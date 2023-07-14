from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_unique_list, get_scalars_list
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.sparse_utils import sparse_matrix

import scipy.sparse as sp
import numpy as np


def get_dot_lite(op):
    if len(op.dependencies[0].shape) == 1:
        return VecDotLite
    else:
        return TenDotLite


class VecDotLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'dot_product'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.in_names = [var.name for var in operation.dependencies]
        self.out_name = operation.outs[0].name
        self.in_shape = operation.dependencies[0].shape[0]
        # self.in_vals = [var.val for var in operation.dependencies]
        self.in1 = self.get_input_id(self.in_names[0])
        self.in2 = self.get_input_id(self.in_names[1])

    def get_evaluation(self, eval_block, vars):

        out = self.get_output_id(self.out_name)
        eval_block.write(f'{out} = np.dot({self.in1}, {self.in2})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac, lazy):

        for key_tuple in partials_dict:
            input_not = key_tuple[1].id
            output = key_tuple[0].id
            partial_name = partials_dict[key_tuple]['name']

            in_size = np.prod(self.in_shape)
            if input_not == self.in1:
                input = self.in2
            else:
                input = self.in1
            partials_block.write(f'{partial_name} = {input}.reshape((1,{in_size}))')


class TenDotLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'tensor_dot_product'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.in_names = [var.name for var in operation.dependencies]
        self.out_name = operation.outs[0].name
        self.in_shape = operation.dependencies[0].shape
        self.axis = operation.literals['axis']
        self.out_shape = operation.outs[0].shape
        # self.in_vals = [var.val for var in operation.dependencies]

        if self.out_shape == None:
            self.out_shape = tuple(np.delete(list(self.in_shape), self.axis))

        # This is all for partial derivative computation
        self.in_size = np.prod(self.in_shape)
        self.out_size = np.prod(self.out_shape)

        self.cols = np.arange(self.in_size)

        self.rows = np.unravel_index(np.arange(self.in_size), shape=self.in_shape)
        self.rows = np.delete(np.array(self.rows), self.axis, axis=0)
        self.rows = np.ravel_multi_index(self.rows, dims=self.out_shape)

        self.in1 = self.get_input_id(self.in_names[0])
        self.in2 = self.get_input_id(self.in_names[1])

    def get_evaluation(self, eval_block, vars):

        out = self.get_output_id(self.out_name)
        eval_block.write(f'{out} = np.sum({self.in1} * {self.in2}, axis={self.axis})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac, lazy):

        row_name = f'{self.name}_rows'
        col_name = f'{self.name}_cols'
        vars[row_name] = self.rows
        vars[col_name] = self.cols

        for key_tuple in partials_dict:
            input_not = key_tuple[1].id
            output = key_tuple[0].id
            partial_name = partials_dict[key_tuple]['name']

            in_size = self.in_size
            out_size = self.out_size
            if input_not == self.in1:
                input = self.in2
            else:
                input = self.in1

            if not is_sparse_jac:
                partials_block.write(f'{partial_name} = np.zeros(({out_size},{in_size}))')
                partials_block.write(f'{partial_name}[{row_name},{col_name}] = {input}.flatten()')
            else:
                partials_block.write(f'{partial_name} = sp.csc_matrix(({input}.flatten(),({row_name}, {col_name})), shape = ({out_size},{in_size}))')
