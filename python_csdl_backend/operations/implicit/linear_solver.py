
from csdl.solvers.linear.direct import DirectSolver
from csdl.solvers.linear.linear_block_gs import LinearBlockGS
from csdl.solvers.linear.linear_block_jac import LinearBlockJac
from csdl.solvers.linear.linear_runonce import LinearRunOnce
from csdl.solvers.linear.petsc_ksp import PETScKrylov
from csdl.solvers.linear.scipy_iter_solver import ScipyKrylov
import numpy as np
import scipy.sparse as sp
from scipy import linalg


def build_linear_solver(csdl_linear_solver, residuals):

    """
    Creates a function that solves the linear system Ax = b, where A is a matrix, b is a vector, and sparse is a boolean that indicates whether A is sparse or not.
    """

    # if isinstance(csdl_linear_solver, DirectSolver) or (csdl_linear_solver is None):
    if 1:
        def lin_solve_function(A, b, sparse):
            """
            Direct Solver
            """
            # return linalg.solve(A, b)

            # print(A.shape,b.shape)
            # print(type(A),type(b), sparse)
            # print(A,b, sparse)
            if sparse:
                return sp.linalg.spsolve(A, b)
            else:
                return linalg.solve(A, b)
            
    elif isinstance(csdl_linear_solver, ScipyKrylov):
        def lin_solve_function(A, b, sparse):
            """
            Scipy's GMRES Solver

            Solves one column at a time if b is a matrix.
            Therefore, we must solve each column of b separately.
            """

            if sp.issparse(b):
                b = b.toarray()

            if len(b.shape) == 1:
                return sp.linalg.gmres(A, b)[0]
            else:
                b_out = np.zeros(b.shape)
                num_cols = b.shape[1]
                for i in range(num_cols):
                    b_out[:, i] = sp.linalg.gmres(
                        A,
                        b[:, i],
                        restart = csdl_linear_solver.options['restart'],
                        maxiter = csdl_linear_solver.options['maxiter'],
                        atol = csdl_linear_solver.options['atol'])[0]
                    
                return b_out

    elif isinstance(csdl_linear_solver, PETScKrylov):
        raise_solver_not_implemented_error(PETScKrylov)
    elif isinstance(csdl_linear_solver, LinearBlockGS):
        raise_solver_not_implemented_error(LinearBlockGS)
    elif isinstance(csdl_linear_solver, LinearBlockJac):
        raise_solver_not_implemented_error(LinearBlockJac)
    elif isinstance(csdl_linear_solver, LinearRunOnce):
        raise_solver_not_implemented_error(LinearRunOnce)
    else:
        raise ValueError(f'linear solver of type {type(csdl_linear_solver)} (for operation with residuals {[r for r in residuals.keys()]}) is not supported.')

    return lin_solve_function

def raise_solver_not_implemented_error(solver):
    raise NotImplementedError(f'linear solver of type {type(solver)} is not yet supported.')