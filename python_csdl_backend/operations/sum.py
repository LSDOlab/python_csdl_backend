from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_list, get_scalars_list
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.sparse_utils import sparse_matrix

import scipy.sparse as sp
import numpy as np


def get_sum_lite(op):

    if op.literals['axes'] is None:
        if len(op.dependencies) == 1:
            return SingleTensorSumCompLite
        else:
            raise NotImplementedError('zero axis multi not implemented')
    else:
        if len(op.dependencies) == 1:
            return SingleTensorSumCompAxisLite
        else:
            raise NotImplementedError('multi axis multi not implemented')


class SingleTensorSumCompLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'single_tensor_sum_no_axis'
        name = f'{name} {op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.in_name = self.get_input_id(operation.dependencies[0].name)
        self.shape = operation.dependencies[0].shape
        self.out_name = self.get_output_id(operation.outs[0].name)
        self.val = operation.dependencies[0].val
        self.out_shape = operation.outs[0].shape

        self.input_size = np.prod(self.shape)

    def get_evaluation(self, eval_block, vars):

        eval_block.write(f'{self.out_name} = np.sum({self.in_name})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):

        key_tuple = get_only(partials_dict)
        input = key_tuple[1]
        output = key_tuple[0]
        partial_name = partials_dict[key_tuple]['name']

        if not is_sparse_jac:
            vars[partial_name] = np.ones((1, self.input_size))
        else:
            vars[partial_name] = np.ones((1, self.input_size))


class SingleTensorSumCompAxisLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'single_tensor_sum_with_axis'
        name = f'{name} {op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.in_name = self.get_input_id(operation.dependencies[0].name)
        self.shape = operation.dependencies[0].shape
        self.out_name = self.get_output_id(operation.outs[0].name)
        self.val = operation.dependencies[0].val
        self.out_shape_true = operation.outs[0].shape
        self.axes = operation.literals['axes']

        self.input_size = np.prod(self.shape)
        self.val = np.ones(self.input_size)
        output_shape = np.delete(self.shape, self.axes)
        self.output_shape = tuple(output_shape)

        self.output_size = np.prod(self.output_shape)

    def get_evaluation(self, eval_block, vars):

        eval_block.write(f'{self.out_name} = np.sum({self.in_name}, axis = {self.axes}).reshape({self.out_shape_true})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):

        key_tuple = get_only(partials_dict)
        input = key_tuple[1]
        output = key_tuple[0]
        partial_name = partials_dict[key_tuple]['name']

        cols = np.arange(self.input_size)

        rows = np.unravel_index(np.arange(self.input_size), shape=self.shape)
        rows = np.delete(np.array(rows), self.axes, axis=0)
        rows = np.ravel_multi_index(rows, dims=self.output_shape)

        if not is_sparse_jac:
            vars[partial_name] = sp.csc_matrix((self.val, (rows, cols)), shape=(self.output_size, self.input_size)).toarray()
        else:
            vars[partial_name] = sp.csc_matrix((self.val, (rows, cols)), shape=(self.output_size, self.input_size))
