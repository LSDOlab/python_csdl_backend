from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_unique_list, get_scalars_list
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
import numpy as np
import scipy.sparse as sp


def get_outer_lite(op):
    if len(op.dependencies[0].shape) == 1 and len(op.dependencies[1].shape) == 1:
        return VectorOuterProductLite
    else:
        return TensorOuterProductLite


class VectorOuterProductLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'vector_outer'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        in_names = [var.name for var in operation.dependencies]
        out_name = operation.outs[0].name
        in_shapes = [var.shape[0] for var in operation.dependencies]
        self.in_shapes = in_shapes
        # in_vals = [var.val for var in operation.dependencies]

        self.out_size = np.prod(operation.outs[0].shape)

        self.rows = np.arange(in_shapes[0] * in_shapes[1])
        self.cols0 = np.repeat(np.arange(in_shapes[0]), in_shapes[1])
        self.cols1 = np.tile(np.arange(in_shapes[1]), in_shapes[0])

        self.out_id = self.get_output_id(out_name)
        self.in_ids = [self.get_input_id(in_name) for in_name in in_names]

    def get_evaluation(self, eval_block, vars):

        eval_block.write(f'{self.out_id} = np.outer({self.in_ids[0]}, {self.in_ids[1]})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac, lazy):

        row_name = f'{self.name}_rows'
        vars[row_name] = self.rows

        for key_tuple in partials_dict:
            input = key_tuple[1].id
            output = key_tuple[0].id
            partial_name = partials_dict[key_tuple]['name']

            col_name = f'{self.name}_{input}_cols'
            if input == self.in_ids[0]:
                line_str = f'val = np.tile({self.in_ids[1]}, {self.in_shapes[0]}).flatten()'
                vars[col_name] = self.cols0
                in_size = self.in_shapes[0]
            if input == self.in_ids[1]:
                line_str = f'val = np.repeat({self.in_ids[0]}, {self.in_shapes[1]}).flatten()'
                vars[col_name] = self.cols1
                in_size = self.in_shapes[1]

            partials_block.write(f'{line_str}')
            if is_sparse_jac:
                partials_block.write(f'{partial_name} = sp.csc_matrix((val ,({row_name},{col_name})), shape = ({self.out_size},{in_size}))')
            else:
                partials_block.write(f'{partial_name} = np.zeros(({self.out_size},{in_size}))')
                partials_block.write(f'{partial_name}[{row_name},{col_name}] = val')


class TensorOuterProductLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'vector_outer'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        in_names = [var.name for var in operation.dependencies]
        out_name = operation.outs[0].name
        in_shapes = [var.shape for var in operation.dependencies]
        self.in_shapes = in_shapes
        # in_vals = [var.val for var in operation.dependencies]

        self.out_size = np.prod(operation.outs[0].shape)
        self.out_id = self.get_output_id(out_name)
        self.in_ids = [self.get_input_id(in_name) for in_name in in_names]

        alphabets = 'abcdefghijklmnopqrstuvwxyz'
        rank0 = len(in_shapes[0])
        rank1 = len(in_shapes[1])
        out_rank = rank0 + rank1
        in0 = alphabets[:rank0]
        in1 = alphabets[rank0:out_rank]
        out = alphabets[:out_rank]
        self.subscript = '{},{}->{}'.format(in0, in1, out)

        self.out_shape = tuple(list(in_shapes[0]) + list(in_shapes[1]))

        out_size = np.prod(self.out_shape)
        self.in_size0 = np.prod(in_shapes[0])
        self.in_size1 = np.prod(in_shapes[1])

        self.rows = np.arange(out_size)
        self.cols0 = np.repeat(np.arange(self.in_size0), self.in_size1)
        self.cols1 = np.tile(np.arange(self.in_size1), self.in_size0)

    def get_evaluation(self, eval_block, vars):

        # eval_block.write(f'{self.out_id} = np.outer({self.in_ids[0]}, {self.in_ids[1]})')
        eval_block.write(f'{self.out_id} = np.einsum(\'{self.subscript}\', {self.in_ids[0]},{self.in_ids[1]}).reshape({self.operation.outs[0].shape})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac, lazy):

        row_name = f'{self.name}_rows'
        vars[row_name] = self.rows

        for key_tuple in partials_dict:
            input = key_tuple[1].id
            output = key_tuple[0].id
            partial_name = partials_dict[key_tuple]['name']

            col_name = f'{self.name}_{input}_cols'
            if input == self.in_ids[0]:
                line_str = f'val = np.tile({self.in_ids[1]}, {self.in_size0}).flatten()'
                vars[col_name] = self.cols0
                in_size = self.in_size0
            if input == self.in_ids[1]:
                line_str = f'val = np.repeat({self.in_ids[0]}, {self.in_size1}).flatten()'
                vars[col_name] = self.cols1
                in_size = self.in_size1

            partials_block.write(f'{line_str}')
            if is_sparse_jac:
                partials_block.write(f'{partial_name} = sp.csc_matrix((val ,({row_name},{col_name})), shape = ({self.out_size},{in_size}))')
            else:
                partials_block.write(f'{partial_name} = np.zeros(({self.out_size},{in_size}))')
                partials_block.write(f'{partial_name}[{row_name},{col_name}] = val')
