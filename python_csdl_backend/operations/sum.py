from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_list, get_scalars_list
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.sparse_utils import sparse_matrix, get_sparsity, SPARSITY_CUTOFF

import scipy.sparse as sp
import numpy as np


def get_sum_lite(op):

    if op.literals['axes'] is None:
        if len(op.dependencies) == 1:
            return SingleTensorSumCompLite
        else:
            return MultipleTensorSumCompLite
    else:
        if len(op.dependencies) == 1:
            return SingleTensorSumCompAxisLite
        else:
            return MultipleTensorSumCompAxisLite


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

        eval_block.write(f'{self.out_name} = np.sum({self.in_name}).reshape({self.out_shape})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):

        key_tuple = get_only(partials_dict)
        partial_name = partials_dict[key_tuple]['name']

        if not is_sparse_jac:
            vars[partial_name] = np.ones((1, self.input_size))
        else:
            vars[partial_name] = np.ones((1, self.input_size))

    def determine_sparse(self):
        return self.determine_sparse_default_elementwise(self.input_size)

class MultipleTensorSumCompLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'multiple_tensor_sum_no_axis'
        name = f'{name} {op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)
        self.elementwise = True

        self.in_names = [self.get_input_id(var.name) for var in operation.dependencies]
        self.shape = operation.dependencies[0].shape
        self.out_name = self.get_output_id(operation.outs[0].name)
        self.out_shape = operation.outs[0].shape
        self.axes = operation.literals['axes']
        self.vals = [var.val for var in operation.dependencies]

        self.input_size = np.prod(self.shape)

    def get_evaluation(self, eval_block, vars):

        eval_block.write(f'{self.out_name} = {self.in_names[0]}')
        for i, in_name in enumerate(self.in_names):
            if i == 0:
                continue
            eval_block.write(f'+ {in_name}', linebreak=False)

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):

        for key_tuple in partials_dict:
            partial_name = partials_dict[key_tuple]['name']

            # if elementwise, return only the diagonal values
            if not is_sparse_jac:
                vars[partial_name] = np.ones((self.input_size,))
            else:
                vars[partial_name] = np.ones((self.input_size,))

    def determine_sparse(self):
        return self.determine_sparse_default_elementwise(self.input_size)

class SingleTensorSumCompAxisLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'single_tensor_sum_with_axis'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.in_name = self.get_input_id(operation.dependencies[0].name)
        self.shape = operation.dependencies[0].shape
        self.out_name = self.get_output_id(operation.outs[0].name)
        self.val = operation.dependencies[0].val
        self.out_shape_true = operation.outs[0].shape
        self.axes = operation.literals['axes']

        self.input_size = np.prod(self.shape)
        self.val = np.ones(self.input_size)
        # output_shape = np.delete(self.shape, self.axes)
        self.output_shape = self.out_shape_true

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
        rows = np.array(rows)
        if len(self.shape) > 1:
            rows = np.delete(np.array(rows), self.axes, axis=0)
            rows = np.ravel_multi_index(rows, dims=self.output_shape)
        else:
            rows = np.zeros(len(cols))

        if not is_sparse_jac:
            vars[partial_name] = sp.csc_matrix((self.val, (rows, cols)), shape=(self.output_size, self.input_size)).toarray()
        else:
            vars[partial_name] = sp.csc_matrix((self.val, (rows, cols)), shape=(self.output_size, self.input_size))

    def determine_sparse(self):
        return self.determine_sparse_default_elementwise(self.input_size)

class MultipleTensorSumCompAxisLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'multiple_tensor_sum_with_axis'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.in_names = [var.name for var in operation.dependencies]
        self.shape = operation.dependencies[0].shape
        self.out_name = operation.outs[0].name
        self.out_shape = operation.outs[0].shape
        self.axes = operation.literals['axes']
        self.vals = [var.val for var in operation.dependencies]

        output_shape = np.delete(self.shape, self.axes)
        self.output_shape = tuple(output_shape)
        self.out_size = np.prod(self.output_shape)
        self.num_inputs = len(self.in_names)
        self.input_size = np.prod(self.shape)
        self.val = np.ones(self.input_size)

        self.cols = np.arange(self.input_size)

        self.rows = np.unravel_index(np.arange(self.input_size), shape=self.shape)
        self.rows = np.delete(np.array(self.rows), self.axes, axis=0)
        self.rows = np.ravel_multi_index(self.rows, dims=self.output_shape)

        self.out_id = self.get_output_id(self.out_name)
        self.in_ids = [self.get_input_id(in_name) for in_name in self.in_names]

    def get_evaluation(self, eval_block, vars):
        
        eval_block.write(f'{self.out_id} = np.sum({self.in_ids[0]}, axis={self.axes})')
        for i in range(1, self.num_inputs):
            eval_block.write(f'{self.out_id} += np.sum({self.in_ids[i]}, axis={self.axes})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):

        for key_tuple in partials_dict:
            partial_name = partials_dict[key_tuple]['name']

            if is_sparse_jac:
                vars[partial_name] = sp.csc_matrix((self.val, (self.rows, self.cols)), shape = (self.out_size, self.input_size))
            else:
                vars[partial_name] = np.zeros((self.out_size, self.input_size))
                vars[partial_name][self.rows, self.cols] = self.val

    def determine_sparse(self):

        if self.input_size < 100:
            return False
        
        if get_sparsity(len(self.val), self.out_size, self.input_size) < SPARSE_SIZE_CUTOFF:
            return True
        return False

    def determine_sparse(self):
        return self.determine_sparse_default_elementwise(self.input_size)