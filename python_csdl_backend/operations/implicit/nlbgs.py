import numpy as np
from scipy import linalg
import scipy.sparse as sp
from python_csdl_backend.operations.implicit.implicit_solver import ImplicitSolverBase
from python_csdl_backend.utils.operation_utils import nl_solver_completion_status
import warnings


class NLBGSSolver(ImplicitSolverBase):

    def __init__(self, op, ins, outs):

        # Solver for NLBGS
        # simple explanation: https: //www.youtube.com/watch?v=pJG4yhtgerg
        # TODO: We can find the order of which states to iterate first by sorting
        # the graph by which residual is "first"
        super().__init__(op, ins, outs)

        self.maxiter = op.nonlinear_solver.options['maxiter']
        self.tol = op.nonlinear_solver.options['atol']

    def _solve_implicit(self):

        # I need to perform the following in this method:
        # bring sim to the solved state.

        # perform NLBGS iterations:
        # while not converged:
        #    x0_new = x0_old - r0(x0_old, x1_old, ... xn_old)
        #    x1_new = x1_old - r1(x0_new, x1_old, ... xn_old)
        #    ...
        #    xn_new = xn_old - rn(x0_new, x1_new, ... xn_old)

        iter = 0

        while True:
            # loop through all residuals
            for state_name in self.states:
                residual_name = self.states[state_name]['residual']
                # compute residuals
                self.function_wrapper.run()

                # compute xi_new
                old_state = self.function_wrapper.get_state(state_name)
                current_residual = self.function_wrapper.get_residual(residual_name)
                new_state = old_state - current_residual
                # print(f'{state_name}, {new_state}')

                # set new states
                self.function_wrapper.set_state(state_name, new_state)

            # compute residual
            self.function_wrapper.run()

            converged = True
            for residual_name in self.residuals:

                # get residual and respective error scalar
                residual_val = self.function_wrapper.get_residual(residual_name)

                error = np.linalg.norm(residual_val.flatten())
                # print(f'iteration {iter}, {residual_name} error: {error}')

                # if any of the residuals do not meet tolerance, no need to compute errors for other residuals
                if error > self.tol:
                    converged = False
                    break

            # if solved or maxiter, end loop
            if converged:
                break
            iter += 1
            if iter >= self.maxiter:
                break

        # print status
        print(nl_solver_completion_status('NLBGS solver', iter, self.tol, converged))
