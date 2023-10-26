from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_unique_list, get_scalars_list
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
import numpy as np
import scipy.sparse as sp
from python_csdl_backend.operations.linear.linear_solver import build_linear_solver

def get_solve_linear_lite(op):
    return SolveLinearLite


class SolveLinearLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'solve_linear_system'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.jac_is_function = True

        # variable names
        out_name = operation.outs[0].name
        self.out_id = self.get_output_id(out_name)
        self.prepend_name = operation.name+self.out_id
        self.ordered_out_names = [out_name]
        self.ordered_in_names = []
        self.ordered_args = []

        # sizes
        self.n = operation.literals['n']

        # A
        self.A_is_var = operation.literals['A_info']['var']
        self.A_obj = operation.literals['A_info']['obj']
        self.A_name = operation.literals['A_info']['name']
        self.sparse = False
        if not self.A_is_var:
            self.A_name = self.prepend_name+'_A'

            if sp.issparse(self.A_obj):
                self.sparse = True
        else:
            self.ordered_in_names.append(self.A_name)
            self.ordered_args.append('A')
            self.A_name = self.get_input_id(self.A_name)
        # b
        self.b_is_var = operation.literals['b_info']['var']
        self.b_obj = operation.literals['b_info']['obj']
        self.b_name = operation.literals['b_info']['name']
        if not self.b_is_var:
            self.b_name = self.prepend_name+'_b'
        else:
            self.ordered_in_names.append(self.b_name)
            self.ordered_args.append('b')
            self.b_name = self.get_input_id(self.b_name)

        # solver
        self.solver = build_linear_solver(operation.literals['solver'], None)
        self.solver_name = self.prepend_name+'_solver'

    def get_evaluation(self, eval_block, vars):
        vars[self.solver_name] = self.solver

        if not self.A_is_var:
            vars[self.A_name] = self.A_obj
        if not self.b_is_var:
            vars[self.b_name] = self.b_obj

        eval_block.write(f'{self.out_id} = {self.solver_name}({self.A_name}, {self.b_name}, {self.sparse})')


    def get_accumulation_function(self, input_paths, path_output, partials_block, vars):
        # Here we generate code to continue jacobian accumulation given accumulated paths from output to this implicit operation
        if not self.A_is_var:
            vars[self.A_name] = self.A_obj
        if not self.b_is_var:
            vars[self.b_name] = self.b_obj

        # implicit solver object
        name_transposed_system = self.prepend_name+'_solve_transposed_system'
        vars[name_transposed_system] = self.accumulate_transpose

        # Ouput paths must be in correct order...
        name_transposed_path = self.prepend_name+'_path_tranposed'
        partials_block.write(f'{name_transposed_path} = {name_transposed_system}({self.A_name},{path_output[0]})')

        # Input paths must be in correct order...
        for i, path in enumerate(input_paths):
            input_arg_type = self.ordered_args[i]

            if input_arg_type == 'A':
                name_accumulate_A = self.prepend_name+'_accumulate_A'
                vars[name_accumulate_A] = self.accumulate_A
                partials_block.write(f'{path} = {name_accumulate_A}({self.out_id},{name_transposed_path})')
            else:
                partials_block.write(f'{path} = {name_transposed_path}.T')

    def accumulate_transpose(self, A, path):
        # print('accumulating transpose', path.T.shape, A.shape)

        path_transpose = path.T
        if sp.issparse(path_transpose) and not self.sparse:
            path_transpose = path_transpose.toarray()
        path_solved_transposed = self.solver(A.T, path_transpose, self.sparse)
        return path_solved_transposed

    def accumulate_A(self, x, path):
        # print('accumulating A', path.shape)
        num_vjps = path.shape[1]
        dA = -np.outer(x,path.T).T.reshape(num_vjps,self.n*self.n)

        return dA