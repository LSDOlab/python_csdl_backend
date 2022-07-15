import numpy as np
from scipy import linalg
from python_csdl_backend.operations.implicit.implicit_solver import ImplicitSolverBase
import scipy.sparse as sp


class SolveResCustom(ImplicitSolverBase):

    def __init__(self, op, ins, outs):
        super().__init__(op, ins, outs)
        self.tol = 1e-10

    def _solve_implicit(self):

        self.function_wrapper.solve_residuals()
