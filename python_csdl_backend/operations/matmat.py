from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_list, get_scalars_list
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
from python_csdl_backend.utils.sparse_utils import get_sparsity, SPARSITY_CUTOFF
import numpy as np


def get_matmat_lite(op):
    return MatmatLite


class MatmatLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'matmat'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.in_names = [var.name for var in operation.dependencies]
        self.out = operation.outs[0].name
        self.out_size = np.prod(operation.outs[0].shape)
        in_shapes = [var.shape for var in operation.dependencies]
        self.in_shapes = in_shapes
        self.input_size = np.prod(in_shapes[0])
        self.in_vals = [var.val for var in operation.dependencies]

        self.input_names = [self.get_input_id(in_name) for in_name in self.in_names]
        self.output_name = self.get_output_id(self.out)

        if in_shapes[1][1] != 1:
            output_shape = (in_shapes[0][0], in_shapes[1][1])

        else:
            output_shape = (in_shapes[0][0], )

        output_size = (np.prod(output_shape))
        input1_size = np.prod(in_shapes[0])
        input2_size = np.prod(in_shapes[1])

        self.r0 = np.repeat(np.arange(output_size), in_shapes[0][1])
        self.c0 = np.tile(
            np.arange(input1_size).reshape(in_shapes[0]),
            in_shapes[1][1]).flatten()

        self.r1 = np.repeat(np.arange(output_size), in_shapes[0][1])
        self.c1 = np.tile(
            np.transpose(np.arange(input2_size).reshape(
                in_shapes[1])).flatten(), in_shapes[0][0])

    def get_evaluation(self, eval_block, vars):

        str = f'{self.output_name} = {self.input_names[0]}@{self.input_names[1]}'
        eval_block.write(str)

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):

        for key_tuple in partials_dict:
            input = key_tuple[1].id
            output = key_tuple[0].id
            partial_name = partials_dict[key_tuple]['name']

            row_name = 'rows'+input+self.name
            col_name = 'cols'+input+self.name
            if self.get_lang_input(input) == self.in_names[0]:
                vars[row_name] = self.r0
                vars[col_name] = self.c0
                in_size = np.prod(self.in_shapes[0])
                other_input = self.get_input_id(self.in_names[1])
                partials_block.write(f'vals = np.tile(np.transpose({other_input}).flatten(), {self.in_shapes[0][0]})')
            elif self.get_lang_input(input) == self.in_names[1]:
                vars[row_name] = self.r1
                vars[col_name] = self.c1
                in_size = np.prod(self.in_shapes[1])
                other_input = self.get_input_id(self.in_names[0])
                partials_block.write(f'vals = np.tile({other_input}, {self.in_shapes[1][1]}).flatten()')

            if is_sparse_jac:
                partials_block.write(f'{partial_name} = sp.csc_matrix((vals, ({row_name},{col_name})), shape = ({self.out_size},{in_size}))')
            else:
                partials_block.write(f'{partial_name} = np.zeros(({self.out_size},{in_size}))')
                partials_block.write(f'{partial_name}[{row_name}, {col_name}] = vals')

    def determine_sparse(self):
        # if self.input_size < 100:
        #     return False

        # if get_sparsity(max(len(self.r0), len(self.r1)), self.out_size, self.input_size) < SPARSITY_CUTOFF:
        #     return True
        # return False

        return self.determine_sparse_default_elementwise(self.input_size)
