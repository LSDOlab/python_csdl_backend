from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_list, get_scalars_list
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
import numpy as np


def get_uqtile_lite(op):
    return UqTileLite


class UqTileLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'uqtile'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.out_name = operation.dependents[0].name
        self.in_name = operation.dependencies[0].name
        self.in_shape = operation.dependencies[0].shape
        self.out_shape = operation.dependents[0].shape
        self.einsum_string = operation.literals['einsum_string']

        self.input_name = self.get_input_id(self.in_name)
        self.output_name = self.get_output_id(self.out_name)

    def get_evaluation(self, eval_block, vars):

        in_name = self.in_name
        out_name = self.out_name
        in_shape = self.in_shape
        out_shape = self.out_shape
        einsum_string = self.einsum_string
        k = out_shape[0]

        # # outputs[out_name] = np.tile(inputs[in_name], tileshape)
        # k = in_shape[0]
        # temp = np.einsum(einsum_string, inputs[in_name], np.ones((k, 1)))
        # # print(temp.shape)
        # outputs[out_name] = np.reshape(temp, out_shape)
        # eval_block.write(f'print({self.input_name}, \'{self.input_name}\')')
        eval_block.write(f'{self.output_name} = np.reshape(np.einsum(\'{einsum_string}\', {self.input_name}, np.ones(({k}, 1))), {out_shape})')
        # eval_block.write(f'print({self.output_name}, \'{self.out_name}\')')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):
        raise ValueError('cannot take derivatives of uq tile operation')
