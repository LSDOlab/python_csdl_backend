import numpy as np
from scipy import linalg
import scipy.sparse as sp
from csdl import CustomImplicitOperation
from python_csdl_backend.operations.implicit.wrappers.implicit_sim_wrapper import ImplicitSimWrapper
from python_csdl_backend.operations.implicit.wrappers.implicit_custom_wrapper import ImplicitCustomWrapper


class ImplicitSolverBase():

    def __init__(self, op, ins, outs):
        """
        solves residual equations for both CustomImplicit and ImplicitOperation types.
        """

        # list of ordered input names and outputs of the implicit function defined by user.
        self.ordered_inputs = ins
        self.ordered_outs = outs

        # build function wrapper attribute.
        # function_wrapper does:
        # - compute the residual values
        # - compute the residual derivatives
        if isinstance(op, CustomImplicitOperation):
            self.function_wrapper = ImplicitCustomWrapper(op, ins, outs)

            if self.function_wrapper.res_inverse_type == 'AIJ':
                self.full_residual_jac = False
            else:
                self.full_residual_jac = True

        else:
            self.function_wrapper = ImplicitSimWrapper(op, ins, outs)
            self.full_residual_jac = True

        # attributes needed by solvers
        self.states = self.function_wrapper.states
        self.total_state_size = self.function_wrapper.total_state_size
        self.residuals = self.function_wrapper.residuals
        self.exposed = self.function_wrapper.exposed
        self.needs_partials = True

    def solve(self, *inputs):
        """
        given input vals in the order of self.ordered_inputs, 
        return output vals in the order of self.ordered_outputs
        """

        # Set inputs to the implicit operation:
        for i, input_name in enumerate(self.ordered_inputs):
            self.function_wrapper.set_input(input_name, inputs[i])

        # Set initial guess:
        for state_name in self.states:
            self.function_wrapper.set_state(state_name, self.states[state_name]['initial_val'])

        # All the initial values are now set. Solve residuals.
        self._solve_implicit()  # method used by subclass
        self.needs_partials = True

        # Set and return solved states and exposed variables
        return_tuple = []
        for output in self.ordered_outs:
            return_tuple.append(self.function_wrapper.get_state(output))
        return_tuple = tuple(return_tuple)
        return return_tuple

    def solve_res_system_rev(self, b):
        """
        given input vals in the order of self.ordered_inputs, 
        return output vals in the order of self.ordered_outputs
        """

        b_built = False
        for state in b:
            if b_built == False:
                num_cols = b[state].shape[0]

                # DENSE/SPARSE
                # b_mat = np.zeros((num_cols, self.total_state_size))
                b_mat = sp.csc_matrix((num_cols, self.total_state_size))

                b_built = True
            il = self.states[state]['index_lower']
            iu = self.states[state]['index_upper']
            b_mat[:, il:iu] = b[state]

        # DENSE/SPARSE
        # x_mat = linalg.solve(self.residual_jac.T, b_mat.T)
        x_mat = sp.linalg.spsolve(self.residual_jac.T, b_mat.T)

        x_mat = x_mat.reshape((b_mat.T).shape)
        x_mat = x_mat.T
        # print('post_transpose', x_mat.shape)

        x_dict = {}
        for state in b:
            il = self.states[state]['index_lower']
            iu = self.states[state]['index_upper']

            if len(x_mat.shape) == 1:
                x_dict[state] = x_mat[il:iu]
            else:
                x_dict[state] = x_mat[:, il:iu]

        return x_dict

    def accumulate_rev(self, *output_paths):
        '''
        is called in adjoint script to compute derivative.
        '''

        b = {}
        # outputs_paths will be in order of ordered_outputs
        for i, out in enumerate(self.ordered_outs):

            # Perform standard adjoint method if the output is a state.
            if out in self.states:
                b[out] = output_paths[i]

        # Need to compute totals for exposed variables first
        if self.full_residual_jac:
            if self.needs_partials:
                self.totals = self.function_wrapper.compute_totals()
                self.build_jac(self.totals)
        # outputs_paths will be in order of ordered_outputs
        for i, exposed in enumerate(self.ordered_outs):
            # If the output is exposed, chain rule the output to the input:
            # py/px = py/pe1 * pe1/px + py/pe12* pe2/px + ... + c_yx
            # c_yx is computed in the loop before this
            if exposed in self.exposed:

                for state in self.states:
                    # TODO: if we know totals(exposed, state) is zero, we can skip this part
                    b[state] += output_paths[i] @ self.totals[(exposed, state)]

        # print(self.ordered_outs)
        # print(b)

        # solve reverse system
        if self.full_residual_jac:
            x = self.solve_res_system_rev(b)
        else:
            if self.function_wrapper.CD_given:
                if self.needs_partials:
                    self.totals = self.function_wrapper.compute_totals()

            x = self.solve_res_system_rev_apply(b)
        self.needs_partials = False
        # print(output_paths)

        # return tuple of outputs in order of ordered_inputs
        accumulated_paths = []
        for input in self.ordered_inputs:
            for i, state in enumerate(self.states):
                res_name = self.states[state]['residual']

                if i == 0:
                    # print(res_name, input, self.totals[(res_name, input)].shape, x[state].shape)

                    jac = x[state]@self.totals[(res_name, input)]
                    # print(res_name, input, jac.shape, x[state].shape, self.totals[(res_name, input)].shape)

                else:
                    # print(res_name, input, jac.shape, x[state].shape, self.totals[(res_name, input)].shape)
                    jac += x[state]@self.totals[(res_name, input)]

            # adjoint has negative
            jac = jac*-1

            for exposed_ind, exposed in enumerate(self.ordered_outs):
                # If the output is exposed, chain rule the output to the input:
                # py/pa = py/pe1 * pe1/pa + py/pe2* pe2/pa + ... + c_xa
                # c_xa is computed before this
                if exposed in self.exposed:
                    # TODO: if we know totals(exposed, input) is zero, we can skip this part
                    jac += output_paths[exposed_ind] @ self.totals[(exposed, input)]

            accumulated_paths.append(jac)

        # paths are now solved
        accumulated_paths = tuple(accumulated_paths)
        return accumulated_paths

    def build_jac(self, totals_dict):
        '''
        Build thhe block matrix jacobian of residuals wrt states given as totals_dict
        '''

        # DENSE/SPARSE
        # self.residual_jac = np.zeros((self.total_state_size, self.total_state_size))
        self.residual_jac = sp.csc_matrix((self.total_state_size, self.total_state_size))

        for tuple_key in totals_dict:

            if (tuple_key[0] not in self.residuals) or (tuple_key[1] not in self.states):
                continue

            state_of = self.residuals[tuple_key[0]]['state']
            state_wrt = tuple_key[1]

            il_o = self.states[state_of]['index_lower']
            iu_o = self.states[state_of]['index_upper']
            il_w = self.states[state_wrt]['index_lower']
            iu_w = self.states[state_wrt]['index_upper']

            self.residual_jac[il_o:iu_o, il_w:iu_w] = totals_dict[tuple_key]

    def _solve_implicit(self):
        """
        replace with non-linear solve method
        """

        raise NotImplementedError('not implemented')

    def solve_res_system_rev_apply(self, b):
        """
        apply inverse jacobian. calls custom implicit operation's apply inverse jacobian.
        should never be called for ImplicitOperations.
        """

        accumulated_paths_rev = {}
        for state_name in b:
            # num_rows should be equal for each out in d_outs
            num_rows = b[state_name].shape[0]

            in_size = self.states[state_name]['size']
            accumulated_paths_rev[state_name] = np.zeros((num_rows, in_size))

        d_out = {}
        for row_ind in range(num_rows):
            for output_name in b:
                shape = self.states[output_name]['shape']
                d_out[output_name] = b[output_name][row_ind, :].reshape(shape)

            solved_rev = self.function_wrapper.apply_inverse_jacT(d_out)

            # Replace each row of initialized path accumulation
            for state_name in b:
                accumulated_paths_rev[state_name][row_ind, :] = solved_rev[state_name].flatten()

        return accumulated_paths_rev
