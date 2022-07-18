from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_list, get_scalars_list
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
import numpy as np


def get_passthrough_lite(op):
    return PassthroughLite


class PassthroughLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'passthrough'
        name = f'{name} {op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.input_name = get_only(self.nx_inputs_dict)
        self.output_name = get_only(self.nx_outputs_dict)
        self.input_size = np.prod(self.nx_inputs_dict[self.input_name].var.shape)

    def get_evaluation(self, eval_block, vars):

        eval_block.write(f'{self.output_name} = {self.input_name}')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):

        key_tuple = get_only(partials_dict)
        input = key_tuple[1]
        output = key_tuple[0]
        partial_name = partials_dict[key_tuple]['name']

        size = self.input_size

        # OLD FULL JACOBIAN
        # if is_sparse_jac:
        #     vars[partial_name] = sp.eye(size, format='csc')
        # else:
        #     vars[partial_name] = np.eye(size)

        # NEW: 
        # only return diag values for elementwise
        # Also sparsity doesn't matter
        vars[partial_name] = np.ones(size)

    def determine_sparse(self):
        return self.determine_sparse_default_elementwise(self.input_size)
