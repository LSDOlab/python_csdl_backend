import numpy as np
from scipy import linalg
from python_csdl_backend.operations.implicit.custom.implicit_solver_custom import ImplicitSolverCustomBase
import scipy.sparse as sp


class NewtonSolverCustom(ImplicitSolverCustomBase):

    def _solve_implicit(self):

        # I have access to:
        # self.state_vals
        # self.residual_vals
        # self.input_vals

        # I need to perform the following in this method:
        # bring sim to the solved state.
        # run self.build_jac on the totals of sim of self.of_list wrt self.wrt_list

        # perform newton iteration:
        # x_new = x_old - (dr/dx_old)^-1 * r(x_old)

        iter = 0
        while True:
            # compute residual
            inputs, outputs, residuals = self.prepare_evaluate_residuals()
            self.op.evaluate_residuals(inputs, outputs, residuals)

            solved = True
            for residual_name, residual_val in residuals.items():

                # get residual and respective error scalar
                error = np.linalg.norm(residual_val.flatten())
                print(f'iteration {iter}, {residual_name} error: {error}')

                # if any of the residuals do not meet tolerance, no need to compute errors for other residuals
                if error > self.tol:
                    solved = False
                    break

            # if solved, end loop
            if solved:
                break
            iter += 1

            # resume Newton iteration:
            # compute jacobian
            # import cProfile
            # profiler = cProfile.Profile()
            # profiler.enable()

            totals = self.prepare_totals()
            self.op.compute_derivatives(inputs, outputs, totals)
            totals = self.process_totals(totals)

            # profiler.disable()
            # profiler.dump_stats('output')
            self.build_jac(totals)

            # get residuals
            res_val = {}
            for residual_name in self.residuals:
                # get residual and respective error scalar
                res_val[self.residuals[residual_name]['state']] = residuals[residual_name].flatten()

            # compute new state
            solved_vec = self.solve_res_system_fwd(res_val)

            # new state
            for state in solved_vec:
                self.state_vals[state] = self.state_vals[state] - solved_vec[state].reshape(self.states[state]['shape'])


    def solve_res_system_fwd(self, b):

        b_vec = np.zeros((self.total_state_size,))
        for state in b:
            il = self.states[state]['index_lower']
            iu = self.states[state]['index_upper']
            b_vec[il:iu] = b[state].flatten()

        # DENSE/SPARSE
        print(self.residual_jac)
        # x_vec = np.linalg.solve(self.residual_jac, b_vec)
        x_vec = sp.linalg.spsolve(self.residual_jac, b_vec)

        x_dict = {}
        for state in b:
            il = self.states[state]['index_lower']
            iu = self.states[state]['index_upper']

            x_dict[state] = x_vec[il:iu]

        return x_dict
