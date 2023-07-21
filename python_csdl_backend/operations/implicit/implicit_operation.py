from csdl import ImplicitOperation, BracketedSearchOperation
from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.state_manager import StateManager

from csdl.solvers.nonlinear.nonlinear_block_gs import NonlinearBlockGS
from csdl.solvers.nonlinear.nonlinear_block_jac import NonlinearBlockJac
from csdl.solvers.nonlinear.nonlinear_runonce import NonlinearRunOnce
from csdl.solvers.nonlinear.newton import NewtonSolver
from csdl.lang.variable import Variable


from python_csdl_backend.operations.implicit.newton import NewtonLiteSolver
from python_csdl_backend.operations.implicit.nlbgs import NLBGSSolver
from python_csdl_backend.operations.implicit.bracket import BracketedSolver
from python_csdl_backend.operations.implicit.solve_residual import SolveResCustom
from csdl import CustomImplicitOperation


def get_implicit_lite(csdl_node):

    # Return an OperationBase object

    # If bracketed search
    if isinstance(csdl_node, BracketedSearchOperation):
        return BracketedSearchLite
    elif not isinstance(csdl_node, ImplicitOperation):
        raise ValueError(f'Implicit operation {csdl_node} is not of type ImplicitOperation or BracketedSearchOperation.')

    # else, return user specified non-linear solver
    nlsolver = csdl_node.nonlinear_solver
    if isinstance(nlsolver, NewtonSolver):
        return NewtonLite
    elif isinstance(nlsolver, NonlinearBlockGS):
        return NonlinearBlockGSLite
    else:
        raise NotImplementedError(f'nonlinear solver {nlsolver} is not yet implemented in this backend')


def get_implicit_custom_lite(csdl_node):

    test_instance = CustomImplicitOperation()
    if test_instance.solve_residual_equations.__func__ is not csdl_node.solve_residual_equations.__func__:
        return SolveResLite

    nlsolver = csdl_node.nonlinear_solver
    if csdl_node.nonlinear_solver is None:
        csdl_node.nonlinear_solver = NewtonSolver()
        return NewtonLite
    elif isinstance(nlsolver, NewtonSolver):
        return NewtonLite
    elif isinstance(nlsolver, NonlinearBlockGS):
        return NonlinearBlockGSLite
    else:
        raise NotImplementedError(f'nonlinear solver {nlsolver} is not yet implemented in this backend')


class ImplicitLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'implict'
        self.operation_name = name
        name = f'{name}_{op_name}'
        operation.properties = {}
        operation.properties['elementwise'] = False
        operation.properties['linear'] = False
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        # set up user input names in order and user output names in order

        temp = [d.name for d in self.operation.dependencies]
        # print(temp)
        self.ordered_in_brackets = {}
        if isinstance(operation, BracketedSearchOperation):
            for state, (l, u) in operation.brackets.items():
                if isinstance(l, Variable):
                    self.ordered_in_brackets[l.name] = {}
                    self.ordered_in_brackets[l.name]['id'] = self.get_input_id(l.name)
                    self.ordered_in_brackets[l.name]['state'] = state
                    self.ordered_in_brackets[l.name]['lower_upper_ind'] = 0

                if isinstance(u, Variable):
                    self.ordered_in_brackets[u.name] = {}
                    self.ordered_in_brackets[u.name]['id'] = self.get_input_id(u.name)
                    self.ordered_in_brackets[u.name]['state'] = state
                    self.ordered_in_brackets[u.name]['lower_upper_ind'] = 1

        # this block here is due to a csdl bug
        self.ordered_in_names = []  # inputs
        for name in temp:
            if (name in self.ordered_in_brackets):
                continue
            if (name not in self.ordered_in_names):
                self.ordered_in_names.append(name)
        # this block here is due to a csdl bug.
        # why are op.outs and op.dependents empty for custom imp ops?
        self.ordered_out_names = [d.name for d in self.operation.outs]  # outputs
        if len(self.ordered_out_names) == 0:
            self.ordered_out_names = [nx_outputs[d].var.name for d in nx_outputs]

        # For all implicit operations, the jacobian is a function.
        # In the future, determine if inverting matrix is viable?
        self.jac_is_function = True

        # The implicit solver defined by subclass
        # -- Newton
        # -- Bracketed
        # -- NLBGS (not yet implemented)
        self.solver = None

        # mapping from state output id to intial state guess name
        self.state_outid_to_initial_guess = {}

    def get_evaluation(self, eval_block, vars):
        # Here we generate code to solve for outputs of implicit operation

        vars[self.operation_name] = self.solver

        # Input variables in order
        in_argument = ''
        for inv in self.ordered_in_names:
            inv_id = self.get_input_id(inv)
            in_argument += inv_id+', '
        for inv in self.ordered_in_brackets:
            inv_id = self.get_input_id(inv)
            in_argument += inv_id+', '
        if in_argument != '':
            in_argument = in_argument.rstrip(in_argument[-1])
            in_argument = in_argument.rstrip(in_argument[-1])

        initial_guesses_arg = ''
        for i, state_name in enumerate(self.solver.states):
            state_guess = self.to_initial_guess_name(state_name)
            initial_guesses_arg += state_guess+', '
        if initial_guesses_arg != '':
            initial_guesses_arg = initial_guesses_arg.rstrip(initial_guesses_arg[-1])
            initial_guesses_arg = initial_guesses_arg.rstrip(initial_guesses_arg[-1])

        # solved_outputs = newton_op(inputs)
        eval_block.write(f'{self.operation_name}.set_guess({initial_guesses_arg})')
        eval_block.write(f'{self.operation_name}_out = {self.operation_name}.solve({in_argument})')

        for i, outv in enumerate(self.ordered_out_names):

            outv_id = self.get_output_id(outv)
            eval_block.write(f'{outv_id} = {self.operation_name}_out[{i}]')

    def get_accumulation_function(self, input_paths, path_output, partials_block, vars):
        # Here we generate code to continue jacobian accumulation given accumulated paths from output to this implicit operation

        # implicit solver object
        vars[self.operation_name] = self.solver

        # Ouput paths must be in correct order...
        in_argument = ''
        for inv in path_output:
            in_argument += inv+', '
        in_argument = in_argument.rstrip(in_argument[-1])
        in_argument = in_argument.rstrip(in_argument[-1])

        # give paths of outputs to inputs
        partials_block.write(f'{self.operation_name}_path_in = {self.operation_name}.accumulate_rev({in_argument})')

        # Input paths must be in correct order...
        for i, path in enumerate(input_paths):
            partials_block.write(f'{path} = {self.operation_name}_path_in[{i}]')

    def set_initial_state_guess(self, state_dict: StateManager):

        for state_name in self.solver.states:
            self.state_outid_to_initial_guess[self.get_output_id(state_name)] = self.to_initial_guess_name(state_name)
            state_dict.reserve_state(self.to_initial_guess_name(state_name), shape = self.solver.states[state_name]['initial_val'].shape)
            state_dict[self.to_initial_guess_name(state_name)] = self.solver.states[state_name]['initial_val']

    def to_initial_guess_name(self, name):
        return f'initial_guess_{self.get_output_id(name)}'


class NewtonLite(ImplicitLite):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        name = f'{name}_newton'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.solver = NewtonLiteSolver(
            operation,
            self.ordered_in_names,
            self.ordered_out_names
        )


class BracketedSearchLite(ImplicitLite):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        name = f'{name}_bracketed'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.solver = BracketedSolver(
            operation,
            self.ordered_in_names,
            self.ordered_out_names,
            self.ordered_in_brackets,
        )


class NonlinearBlockGSLite(ImplicitLite):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        name = f'{name}_nlbgs'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.solver = NLBGSSolver(
            operation,
            self.ordered_in_names,
            self.ordered_out_names,
        )

# class NonlinearBlockJacobiLite(ImplicitLite):

#     def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
#         name = f'{name}_nlbgs'
#         super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

#         self.solver = NLBJacobiSolver(
#             operation,
#             self.ordered_in_names,
#             self.ordered_out_names,
#         )


class SolveResLite(ImplicitLite):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        name = f'{name}_solve_res'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.solver = SolveResCustom(
            operation,
            self.ordered_in_names,
            self.ordered_out_names
        )
