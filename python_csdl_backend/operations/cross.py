from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_list, get_scalars_list
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.sparse_utils import sparse_matrix

import scipy.sparse as sp
import numpy as np

alphabet = 'abcdefghij'


def get_cross_lite(op):
    return CrossLite


class CrossLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'cross_product'
        name = f'{name} {op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.shape = operation.outs[0].shape
        self.in1_name = operation.dependencies[0].name
        self.in2_name = operation.dependencies[1].name
        self.out_name = operation.outs[0].name
        self.axis = operation.literals['axis']
        self.in1_val = operation.dependencies[0].val
        self.in2_val = operation.dependencies[1].val

        self.outsize = np.prod(self.shape)
        self.insize = self.outsize

        indices = get_array_indices(*self.shape)

        self.shape_without_axis = self.shape[:self.axis] + self.shape[self.axis + 1:]

        ones = np.ones(3, int)

        rank = len(self.shape_without_axis)

        einsum_string_rows = '{}y{},z->{}{}yz'.format(
            alphabet[:self.axis],
            alphabet[self.axis:rank],
            alphabet[:self.axis],
            alphabet[self.axis:rank],
        )

        einsum_string_cols = '{}y{},z->{}{}zy'.format(
            alphabet[:self.axis],
            alphabet[self.axis:rank],
            alphabet[:self.axis],
            alphabet[self.axis:rank],
        )

        self.rows1 = np.einsum(
            einsum_string_rows,
            indices,
            ones,
        ).flatten()

        self.cols1 = np.einsum(
            einsum_string_cols,
            indices,
            ones,
        ).flatten()

        # self.declare_partials(out_name, in1_name, rows=rows, cols=cols)

        self.rows2 = np.einsum(
            einsum_string_rows,
            indices,
            ones,
        ).flatten()

        self.cols2 = np.einsum(
            einsum_string_cols,
            indices,
            ones,
        ).flatten()

        # self.declare_partials(out_name, in2_name, rows=rows, cols=cols)

    def get_evaluation(self, eval_block, vars):

        out = self.get_output_id(self.out_name)
        in1 = self.get_input_id(self.in1_name)
        in2 = self.get_input_id(self.in2_name)
        eval_block.write(f'{out} = np.cross({in1}, {in2}, axisa = {self.axis}, axisb = {self.axis}, axisc = {self.axis})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):

        for key_tuple in partials_dict:
            input = key_tuple[1].id
            output = key_tuple[0].id
            partial_name = partials_dict[key_tuple]['name']

            if input == self.get_input_id(self.in1_name):
                # in_name = self.in1_name
                # row_name = f'{in_name}_rows1'
                # col_name = f'{in_name}_cols1'
                # vars[row_name] = self.rows1
                # vars[col_name] = self.cols1

                def compute_cross_jac(input1, input2):
                    ones = np.ones(3)
                    eye = np.eye(3)
                    rank = len(self.shape_without_axis)

                    tmps = {0: None, 1: None, 2: None}
                    for ind in range(3):

                        array = np.einsum(
                            '...,m->...m',
                            np.cross(
                                np.einsum(
                                    '...,m->...m',
                                    np.ones(self.shape_without_axis),
                                    eye[ind, :],
                                ),
                                input2,
                                axisa=-1,
                                axisb=self.axis,
                                axisc=-1,
                            ),
                            eye[ind, :],
                        )

                        tmps[ind] = array

                    val = (tmps[0] + tmps[1] + tmps[2]).flatten()

                    if is_sparse_jac:
                        jac = sp.csc_matrix((val, (self.rows1, self.cols1)), shape=(self.outsize, self.insize))
                    else:
                        jac = np.zeros((self.outsize, self.insize))
                        jac[self.rows1, self.cols1] = val

                    return jac

            elif input == self.get_input_id(self.in2_name):
                # in_name = self.in2_name
                # row_name = f'{in_name}_rows2'
                # col_name = f'{in_name}_cols2'
                # vars[row_name] = self.rows2
                # vars[col_name] = self.cols2

                def compute_cross_jac(input1, input2):
                    eye = np.eye(3)

                    tmps = {0: None, 1: None, 2: None}
                    for ind in range(3):

                        array = np.einsum(
                            '...,m->...m',
                            np.cross(
                                input1,
                                np.einsum(
                                    '...,m->...m',
                                    np.ones(self.shape_without_axis),
                                    eye[ind, :],
                                ),
                                axisa=self.axis,
                                axisb=-1,
                                axisc=-1,
                            ),
                            eye[ind, :],
                        )

                        tmps[ind] = array

                    val = (tmps[0] + tmps[1] + tmps[2]).flatten()

                    if is_sparse_jac:
                        jac = sp.csc_matrix((val, (self.rows2, self.cols2)), shape=(self.outsize, self.insize))
                    else:
                        jac = np.zeros((self.outsize, self.insize))
                        jac[self.rows2, self.cols2] = val

                    return jac

            func_name = self.operation.name+'_'+input+'_jac'
            vars[func_name] = compute_cross_jac

            partials_block.write(f'{partial_name} = {func_name}({self.get_input_id(self.in1_name)}, {self.get_input_id(self.in2_name)})')

    def determine_sparse(self):
        # in1_sparsity = len(self.rows1)/(self.outsize*self.insize)
        # in2_sparsity = len(self.rows2)/(self.outsize*self.insize)

        if (self.outsize > 100) and (self.insize > 100):
            # print(in1_sparsity, in2_sparsity, self.outsize, self.insize)
            return True
        return False


def get_array_indices(*shape):
    return np.arange(np.prod(shape)).reshape(shape)
