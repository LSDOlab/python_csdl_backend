from csdl import ImplicitOperation, BracketedSearchOperation
from python_csdl_backend.operations.operation_base import OperationBase

from csdl.solvers.nonlinear.nonlinear_block_gs import NonlinearBlockGS
from csdl.solvers.nonlinear.nonlinear_block_jac import NonlinearBlockJac
from csdl.solvers.nonlinear.nonlinear_runonce import NonlinearRunOnce
from csdl.solvers.nonlinear.newton import NewtonSolver

# from python_csdl_backend.operations.implicit.nonlinear_block_gs import NLBGSLite
# from python_csdl_backend.operations.implicit.nonlinear_block_jac import NLBJLite
# from python_csdl_backend.operations.implicit.nonlinear_runonce import NLROLite
from python_csdl_backend.operations.implicit.newton import NewtonSolverLite
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
    else:
        raise NotImplementedError(f'nonlinear solver {nlsolver} is not yet implemented in this backend')


def get_implicit_custom_lite(csdl_node):

    test_instance = CustomImplicitOperation()
    if test_instance.solve_residual_equations.__func__ is not csdl_node.solve_residual_equations.__func__:
        return SolveResLite

    nlsolver = csdl_node.nonlinear_solver
    if csdl_node.nonlinear_solver is None:
        return NewtonLite
    else:
        raise NotImplementedError(f'nonlinear solver {nlsolver} is not yet implemented in this backend')


class ImplicitLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'implict'
        self.operation_name = name
        name = f'{name}_{op_name}'
        operation.properties = {}
        operation.properties['elementwise'] = False
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        # set up user input names in order and user output names in order

        temp = [d.name for d in self.operation.dependencies]

        # this block here is due to a csdl bug
        self.ordered_in_names = []  # inputs
        for name in temp:
            if name not in self.ordered_in_names:
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

    def get_evaluation(self, eval_block, vars):
        # Here we generate code to solve for outputs of implicit operation

        vars[self.operation_name] = self.solver

        # Input variables in order
        in_argument = ''
        for inv in self.ordered_in_names:

            inv_id = self.get_input_id(inv)
            in_argument += inv_id+', '
        in_argument = in_argument.rstrip(in_argument[-1])
        in_argument = in_argument.rstrip(in_argument[-1])

        # solved_outputs = newton_op(inputs)
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


class NewtonLite(ImplicitLite):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        name = f'{name}_newton'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.solver = NewtonSolverLite(
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
            self.ordered_out_names
        )


class SolveResLite(ImplicitLite):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        name = f'{name}_solve_res'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.solver = SolveResCustom(
            operation,
            self.ordered_in_names,
            self.ordered_out_names
        )
