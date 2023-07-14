from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_unique_list, get_scalars_list
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
import numpy as np


def get_cotan_lite(op):
    return CotanLite


class CotanLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'cotan'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.shape = self.operation.dependencies[0].shape
        self.input_size = np.prod(self.shape)
        self.in_name = self.operation.dependencies[0].name
        self.out_name = self.operation.outs[0].name
        # self.val = self.operation.dependencies[0].val

        self.input_name = self.get_input_id(self.in_name)
        self.output_name = self.get_output_id(self.out_name)

    def get_evaluation(self, eval_block, vars):

        eval_block.write(f'{self.output_name} = 1.0/np.tan({self.input_name})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac, lazy):

        key_tuple = get_only(partials_dict)
        input = key_tuple[1].id
        output = key_tuple[0].id
        partial_name = partials_dict[key_tuple]['name']

        # OLD FULL JACOBIAN
        inner_str = f'-1.0/(np.sin({input})**2).flatten()'
        # if is_sparse_jac:
        #     partials_block.write(f'{partial_name} = sp.diags({inner_str}, format = \'csc\')')
        # else:
        # partials_block.write(f'{partial_name} = np.diag({inner_str}.flatten())')

        # NEW:
        # only return diag values for elementwise
        # Also sparsity doesn't matter
        partials_block.write(f'{partial_name} = {inner_str}')

    def determine_sparse(self):
        return self.determine_sparse_default_elementwise(self.input_size)
