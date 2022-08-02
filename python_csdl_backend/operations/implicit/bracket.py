import numpy as np
from scipy import linalg
import scipy.sparse as sp
from python_csdl_backend.operations.implicit.implicit_solver import ImplicitSolverBase
from python_csdl_backend.utils.operation_utils import nl_solver_completion_status
import warnings
from csdl.lang.variable import Variable


class BracketedSolver(ImplicitSolverBase):

    def __init__(self, op, ins, outs, bracket_vars):
        bracket_vars_jump = bracket_vars.copy()
        super().__init__(op, ins, outs)

        self.brackets_map = op.brackets
        self.ordered_in_brackets = bracket_vars_jump

        self.max_iter = op.maxiter
        self.max_iter = 100
        self.tol = 1e-12

    def _solve_implicit(self):

        # I have access to:
        # self.sim
        # self.residuals
        # self.build_jac
        # self.of_list
        # self.wrt_list

        # I need to perform the following in this method:
        # bring sim to the solved state.
        # run self.build_jac on the totals of sim of self.of_list wrt self.wrt_list

        # initialize lower and upper state brackets
        x_lower = dict()
        x_upper = dict()
        r_lower = dict()
        r_upper = dict()

        # update bracket for state associated with each residual
        for residual_name in self.residuals:
            state_name = self.residuals[residual_name]['state']
            shape = self.states[state_name]['shape']

            if isinstance(self.brackets_map[state_name][0], np.ndarray):
                if np.prod(self.brackets_map[state_name][0].shape) == np.prod(shape):
                    x_lower[state_name] = self.brackets_map[state_name][0].reshape(shape)
                else:
                    x_lower[state_name] = self.brackets_map[state_name][0] * np.ones(shape)
            else:
                x_lower[state_name] = self.brackets_map[state_name][0] * np.ones(shape)

            if isinstance(self.brackets_map[state_name][1], np.ndarray):
                if np.prod(self.brackets_map[state_name][1].shape) == np.prod(shape):
                    x_upper[state_name] = self.brackets_map[state_name][1].reshape(shape)
                else:
                    x_upper[state_name] = self.brackets_map[state_name][1] * np.ones(shape)
            else:
                x_upper[state_name] = self.brackets_map[state_name][1] * np.ones(shape)

        # Compute residuals at each end of the bracket.
        # Lower:
        for state_name, lower_value in x_lower.items():
            self.function_wrapper.set_state(state_name, lower_value)

        self.function_wrapper.run()

        for residual_name, res_dict in self.residuals.items():
            state_name = res_dict['state']
            r_lower[state_name] = self.function_wrapper.get_residual(residual_name)

        # Upper:
        for state_name, upper_value in x_upper.items():
            self.function_wrapper.set_state(state_name, upper_value)
            # self.sim[state_name] = upper_value

        self.function_wrapper.run()

        for residual_name, res_dict in self.residuals.items():
            state_name = res_dict['state']
            r_upper[state_name] = self.function_wrapper.get_residual(residual_name)

        xp = dict()
        xn = dict()
        # initialize bracket array elements associated with
        # positive and negative residuals so that updates to
        # brackets are associated with a residual of the
        # correct sign from the start of the bracketed search
        for residual_name, res_dict in self.residuals.items():
            state_name = res_dict['state']
            shape = self.states[state_name]['shape']

            mask1 = r_lower[state_name] >= r_upper[state_name]
            mask2 = r_lower[state_name] < r_upper[state_name]

            xp[state_name] = np.empty(shape)
            xp[state_name][mask1] = x_lower[state_name][mask1]
            xp[state_name][mask2] = x_upper[state_name][mask2]

            xn[state_name] = np.empty(shape)
            xn[state_name][mask1] = x_upper[state_name][mask1]
            xn[state_name][mask2] = x_lower[state_name][mask2]

        # Main bisection loop
        x = dict()
        bad_res = (0,'none')
        for iter_num in range(self.max_iter):

            # Compute midpoint of upper and lower bounds
            for residual_name, res_dict in self.residuals.items():
                state_name = res_dict['state']
                x[state_name] = 0.5 * xp[state_name] + 0.5 * xn[state_name]
                self.function_wrapper.set_state(state_name, x[state_name])

            self.function_wrapper.run()

            # Check if residual is within tolerance
            converged = True
            for residual_name, res_dict in self.residuals.items():
                res_norm = np.linalg.norm(self.function_wrapper.get_residual(residual_name))
                # print(f'ITERATION {iter_num} {residual_name}: {res_norm}')

                if res_norm >= self.tol:
                    bad_res = (res_norm, residual_name)
                    converged = False
            if converged:
                # break loop if all residuals are sufficiently small
                break

            # get new residual bracket values
            for residual_name, res_dict in self.residuals.items():
                state_name = res_dict['state']

                # make sure bracket always contains r == 0
                mask_p = self.function_wrapper.get_residual(residual_name) >= 0
                mask_n = self.function_wrapper.get_residual(residual_name) < 0
                xp[state_name][mask_p] = x[state_name][mask_p]
                xn[state_name][mask_n] = x[state_name][mask_n]

        # solver terminates:
        # if not converged:
        #     warnings.warn(f'nonlinear solver: bracketed search did not converge in {self.max_iter} iterations.')

        # print status of nlsolver
        print(nl_solver_completion_status('bracketed search', iter_num, self.tol, converged))
        if not converged:
            print('norm', bad_res[0], '\tname', bad_res[1])

        # for residual_name, res_dict in self.residuals.items():
        #     state_name = res_dict['state']
        #     x[state_name] = 0.5 * xp[state_name] + 0.5 * xn[state_name]
        #     self.sim[state_name] = x[state_name]
        # self.sim.run()

        # if self.full_residual_jac:
        #     self.totals = self.function_wrapper.compute_totals()
        #     self.build_jac(self.totals)
