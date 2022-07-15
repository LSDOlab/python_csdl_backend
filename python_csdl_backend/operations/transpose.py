from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_list, get_scalars_list
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
import numpy as np
import scipy.sparse as sp


def get_transpose_lite(op):
    return TransposeLite


class TransposeLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'transpose'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.shape = self.operation.dependencies[0].shape
        in_shape = self.shape
        self.input_size = np.prod(self.shape)
        size = self.input_size
        in_name = self.operation.dependencies[0].name
        self.out_name = self.operation.outs[0].name
        val = self.operation.dependencies[0].val
        out_shape = self.operation.outs[0].shape,

        self.input_name = self.get_input_id(in_name)
        self.output_name = self.get_output_id(self.out_name)

        self.rank = len(in_shape)

        # self.add_input(in_name, shape=in_shape, val=val)

        if out_shape == None:
            out_shape = in_shape[::-1]

        # self.add_output(out_name, shape=out_shape)

        out_size = np.prod(out_shape)
        size = np.prod(in_shape)

        rows = np.arange(size)
        initial_locations = np.arange(size).reshape(in_shape)
        new_locations = np.transpose(initial_locations)
        cols = new_locations.flatten()

        # Alternate method
        # ================

        # cols = np.arange(size)
        # initial_locations = np.unravel_index(np.arange(size), shape = in_shape)
        # new_locations = np.array(initial_locations)[::-1, :]
        # rows = np.ravel_multi_index(new_locations, dims = out_shape)

        val = np.ones((size, ))

        self.deriv = sp.csc_matrix((val, (rows, cols)), shape=(out_size, size))
        # self.declare_partials(out_name, in_name, rows=rows, cols=cols, val=val)

    def get_evaluation(self, eval_block, vars):

        eval_block.write(f'{self.output_name} = np.transpose({self.input_name})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):

        key_tuple = get_only(partials_dict)
        input = key_tuple[1].id
        output = key_tuple[0].id
        partial_name = partials_dict[key_tuple]['name']

        if is_sparse_jac:
            vars[partial_name] = self.deriv
        else:
            vars[partial_name] = self.deriv.toarray()
