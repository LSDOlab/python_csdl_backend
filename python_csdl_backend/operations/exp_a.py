from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_list, get_scalars_list
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF

import numpy as np


def get_exp_a_lite(op):
    return ExpALite


class ExpALite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'exp_a'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        input_name_id = get_only(self.nx_inputs_dict)
        output_name_id = get_only(self.nx_outputs_dict)
        self.input_size = np.prod(self.nx_inputs_dict[input_name_id].var.shape)

        self.a = operation.literals['a']

        self.input_name = input_name_id
        self.output_name = output_name_id

        self.a_name = self.name + '_a'

    def get_evaluation(self, eval_block, vars):

        vars[self.a_name] = self.a
        eval_block.write(f'{self.output_name} = {self.a_name}**{self.input_name}')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):

        key_tuple = get_only(partials_dict)
        input = key_tuple[1].id
        output = key_tuple[0].id
        partial_name = partials_dict[key_tuple]['name']

        # OLD FULL JACOBIAN
        # if is_sparse_jac:
        #     partials_block.write(f'{partial_name} = sp.diags(np.exp_a({input}).flatten(), format = \'csc\')')
        # else:
        #     partials_block.write(f'{partial_name} = np.diag(np.exp_a({input}).flatten())')

        # NEW:
        # only return diag values for elementwise
        # Also sparsity doesn't matter
        vars[self.a_name] = self.a
        partials_block.write(f'{partial_name} = ({self.a_name}**{input} * np.log({self.a_name})).flatten()')

    def determine_sparse(self):
        return self.determine_sparse_default_elementwise(self.input_size)