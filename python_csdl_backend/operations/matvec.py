from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_list, get_scalars_list
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
import numpy as np


def get_matvec_lite(op):

    if op.literals['sparsemtx'] is None:
        return MatvecLite
    else:
        return MatvecSparseLite


class MatvecLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'matvec'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.in_names = [var.name for var in operation.dependencies]
        self.out_name = operation.outs[0].name
        self.out_size = np.prod(operation.outs[0].shape)
        self.in_shapes = [var.shape for var in operation.dependencies]
        self.in_vals = [var.val for var in operation.dependencies]

        self.input_names = [self.get_input_id(in_name) for in_name in self.in_names]
        self.output_name = self.get_output_id(self.out_name)

        self.c = np.arange(np.prod(self.in_shapes[0]))
        self.r = np.repeat(np.arange(self.in_shapes[0][0]), self.in_shapes[0][1])

    def get_evaluation(self, eval_block, vars):

        str = f'{self.output_name} = {self.input_names[0]}@{self.input_names[1]}'
        eval_block.write(str)

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):

        for key_tuple in partials_dict:
            input = key_tuple[1].id
            output = key_tuple[0].id
            partial_name = partials_dict[key_tuple]['name']

            if self.get_lang_input(input) == self.in_names[0]:
                row_name = 'rows'+input+self.name
                col_name = 'cols'+input+self.name
                vars[row_name] = self.r
                vars[col_name] = self.c
                in_size = np.prod(self.in_shapes[0])
                other_input = self.get_input_id(self.in_names[1])
                partials_block.write(f'vals = np.tile({other_input}, {self.in_shapes[0][0]})')

                if is_sparse_jac:
                    partials_block.write(f'{partial_name} = sp.csc_matrix((vals, ({row_name},{col_name})), shape = ({self.out_size},{in_size}))')
                else:
                    partials_block.write(f'{partial_name} = np.zeros(({self.out_size},{in_size}))')
                    partials_block.write(f'{partial_name}[{row_name}, {col_name}] = vals')

            elif self.get_lang_input(input) == self.in_names[1]:
                other_input = self.get_input_id(self.in_names[0])
                if is_sparse_jac:
                    partials_block.write(f'{partial_name} = sp.csc_matrix({other_input})')
                else:
                    partials_block.write(f'{partial_name} = {other_input}')


class MatvecSparseLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'matvec'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.in_name = operation.dependencies[0].name
        self.out_name = operation.outs[0].name
        self.A = operation.literals['sparsemtx']
        self.in_val = operation.dependencies[0].val

        self.input_name = self.get_input_id(self.in_name)
        self.output_name = self.get_output_id(self.out_name)

    def get_evaluation(self, eval_block, vars):

        a_name = self.name+'_A'
        vars[a_name] = self.A

        str = f'{self.output_name} = {a_name}.dot({self.input_name})'
        eval_block.write(str)

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):

        key_tuple = get_only(partials_dict)
        input = key_tuple[1].id
        output = key_tuple[0].id
        partial_name = partials_dict[key_tuple]['name']

        vars[partial_name] = self.A
