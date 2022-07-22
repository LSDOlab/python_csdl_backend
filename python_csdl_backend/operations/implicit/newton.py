import numpy as np
from scipy import linalg
from python_csdl_backend.operations.implicit.implicit_solver import ImplicitSolverBase
from python_csdl_backend.utils.operation_utils import nl_solver_completion_status
import scipy.sparse as sp


class NewtonSolverLite(ImplicitSolverBase):

    def __init__(self, op, ins, outs):
        super().__init__(op, ins, outs)
        self.tol = 1e-10

    def _solve_implicit(self):

        # I have access to:
        # self.function_wrapper
        # self.residuals
        # self.build_jac

        # I need to perform the following in this method:
        # bring sim to the solved state.
        # run self.build_jac on the totals of sim of self.of_list wrt self.wrt_list

        # perform newton iteration:
        # x_new = x_old - (dr/dx_old)^-1 * r(x_old)

        iter = 0
        while True:
            # compute residual
            self.function_wrapper.run()

            solved = True
            for residual_name in self.residuals:

                # get residual and respective error scalar
                residual_val = self.function_wrapper.get_residual(residual_name)

                error = np.linalg.norm(residual_val.flatten())
                # print(f'iteration {iter}, {residual_name} error: {error}')

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

            if self.full_residual_jac:
                totals = self.function_wrapper.compute_totals()

                # profiler.disable()
                # profiler.dump_stats('output')
                self.build_jac(totals)

                # get residuals
                res_val = {}
                for residual_name in self.residuals:
                    # get residual and respective error scalar
                    # res_val[self.residuals[residual_name]['state']] = self.sim[residual_name].flatten()
                    res_val[self.residuals[residual_name]['state']] = self.function_wrapper.get_residual(residual_name).flatten()

                # compute new state
                solved_vec = self.solve_res_system_fwd(res_val)
            else:
                # get residuals
                res_val = {}
                for residual_name in self.residuals:
                    # get residual and respective error scalar
                    # res_val[self.residuals[residual_name]['state']] = self.sim[residual_name].flatten()
                    res_val[self.residuals[residual_name]['state']] = self.function_wrapper.get_residual(residual_name).flatten()

                if self.function_wrapper.CD_given:
                    self.function_wrapper.compute_totals()

                # compute new state
                solved_vec = self.function_wrapper.apply_inverse_jac(res_val)
            # new state
            for state in solved_vec:
                new_state = self.function_wrapper.get_state(state) - solved_vec[state].reshape(self.states[state]['shape'])
                self.function_wrapper.set_state(state, new_state)

        print(nl_solver_completion_status('newton solver', iter, self.tol, solved))

        # Shouldn't be necessary?
        # if self.full_residual_jac:
        #     self.totals = self.function_wrapper.compute_totals()
        #     self.build_jac(self.totals)

    def solve_res_system_fwd(self, b):

        b_vec = np.zeros((self.total_state_size,))
        for state in b:
            il = self.states[state]['index_lower']
            iu = self.states[state]['index_upper']
            b_vec[il:iu] = b[state].flatten()

        # DENSE/SPARSE
        # x_vec = np.linalg.solve(self.residual_jac, b_vec)
        x_vec = sp.linalg.spsolve(self.residual_jac, b_vec)

        x_dict = {}
        for state in b:
            il = self.states[state]['index_lower']
            iu = self.states[state]['index_upper']

            x_dict[state] = x_vec[il:iu]

        return x_dict
