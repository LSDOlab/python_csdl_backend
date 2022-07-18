from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_list, get_scalars_list
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
import numpy as np
import scipy.sparse as sp


def get_print_var_lite(op):
    return PrintVarLite


class PrintVarLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'print_var'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.in_name = self.operation.dependencies[0].name
        self.out_name = self.operation.dependencies[0].name + '_print'
        self.shape = self.operation.dependencies[0].shape
        self.val = self.operation.dependencies[0].val
        self.input_size = np.prod(self.operation.dependencies[0].shape)

        self.input_name = self.get_input_id(self.in_name)
        # self.output_name = self.get_output_id(self.out_name)

    def get_evaluation(self, eval_block, vars):

        eval_block.write('print()')
        eval_block.write(f'print(\'printing \', \'{self.input_name}\')')
        eval_block.write(f'print({self.input_name})')
        eval_block.write(f'{self.out_name} = {self.input_name}')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):

        key_tuple = get_only(partials_dict)
        input = key_tuple[1].id
        output = key_tuple[0].id
        partial_name = partials_dict[key_tuple]['name']

        # OLD FULL JACOBIAN
        # if is_sparse_jac:
        #     vars[partial_name] = sp.eye(self.input_size, format='csc')
        # else:
        #     vars[partial_name] = np.eye(self.input_size)

        # NEW: 
        # only return diag values for elementwise
        # Also sparsity doesn't matter
        vars[partial_name] = np.ones(self.input_size)

    def determine_sparse(self):
        return self.determine_sparse_default_elementwise(self.input_size)
