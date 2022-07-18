from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_list, get_scalars_list
from python_csdl_backend.utils.general_utils import get_only
import numpy as np


def get_arctan_lite(op):
    return ArctanLite


class ArctanLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'arctan'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        input_name_id = get_only(self.nx_inputs_dict)
        output_name_id = get_only(self.nx_outputs_dict)
        self.input_size = np.prod(self.nx_inputs_dict[input_name_id].var.shape)

        self.input_name = input_name_id
        self.output_name = output_name_id

    def get_evaluation(self, eval_block, vars):

        eval_block.write(f'{self.output_name} = np.arctan({self.input_name})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):

        key_tuple = get_only(partials_dict)
        input = key_tuple[1].id
        output = key_tuple[0].id
        partial_name = partials_dict[key_tuple]['name']

        # # OLD FULL JACOBIAN
        # if is_sparse_jac:
        #     partials_block.write(f'{partial_name} = sp.diags((1.0 / (1.0 + {input}**2)).flatten(), format = \'csc\')')
        # else:
        #     partials_block.write(f'{partial_name} = np.diag((1.0 / (1.0 + {input}**2)).flatten())')

        # NEW: 
        # only return diag values for elementwise
        # Also sparsity doesn't matter
        partials_block.write(f'{partial_name} = (1.0 / (1.0 + {input}**2)).flatten()')

    def determine_sparse(self):
        return self.determine_sparse_default_elementwise(self.input_size)
