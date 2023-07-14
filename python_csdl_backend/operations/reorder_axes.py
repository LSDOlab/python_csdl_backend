from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_unique_list, get_scalars_list
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
import numpy as np
import scipy.sparse as sp


def get_reorder_axes_lite(op):
    return ReorderAxesLite


class ReorderAxesLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'reorder_axes'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        in_name = operation.dependencies[0].name
        in_shape = operation.dependencies[0].shape
        out_name = operation.outs[0].name
        out_shape = operation.outs[0].shape
        operation_literal = operation.literals['operation']
        new_axes_locations = operation.literals['new_axes_locations']
        # val = operation.dependencies[0].val

        self.out_id = self.get_output_id(out_name)
        self.in_id = self.get_input_id(in_name)

        if new_axes_locations == None:
            self.new_axes_locations = compute_new_axes_locations(
                in_shape, operation_literal)
        else:
            self.new_axes_locations = new_axes_locations

        if out_shape == None:
            out_shape = tuple(in_shape[i] for i in self.new_axes_locations)

        size = np.prod(in_shape)
        self.rows = np.arange(size)
        initial_locations = np.arange(size).reshape(in_shape)
        new_locations = np.transpose(initial_locations,
                                     self.new_axes_locations)
        self.cols = new_locations.flatten()
        self.val = np.ones((size, ))

        self.out_size = np.prod(out_shape)
        self.in_size = np.prod(in_shape)
        self.linear = True

    def get_evaluation(self, eval_block, vars):

        new_axes_location_name = f'{self.name}_axes_loc'
        vars[new_axes_location_name] = self.new_axes_locations
        eval_block.write(f'{self.out_id} = np.transpose({self.in_id}, {new_axes_location_name})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac, lazy):

        key_tuple = get_only(partials_dict)
        partial_name = partials_dict[key_tuple]['name']

        if not is_sparse_jac:
            vars[partial_name] = sp.csc_matrix((self.val, (self.rows, self.cols)), shape=(self.out_size, self.in_size)).toarray()
        else:
            vars[partial_name] = sp.csc_matrix((self.val, (self.rows, self.cols)), shape=(self.out_size, self.in_size))

    def determine_sparse(self):
        return self.determine_sparse_default_elementwise(self.in_size)
