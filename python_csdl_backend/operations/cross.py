from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_unique_list, get_scalars_list
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
        self.ordered_in_names = [self.in1_name, self.in2_name]
        self.out_name = operation.outs[0].name
        self.ordered_out_names = [self.out_name]

        self.axis = operation.literals['axis']
        # self.in1_val = operation.dependencies[0].val
        # self.in2_val = operation.dependencies[1].val

        self.outsize = np.prod(self.shape)
        self.insize = self.outsize
        self.shape_without_axis = self.shape[:self.axis] + self.shape[self.axis + 1:]

        self.ones = np.ones(3, int)

        rank = len(self.shape_without_axis)

        self.einsum_string_rows = '{}y{},z->{}{}yz'.format(
            alphabet[:self.axis],
            alphabet[self.axis:rank],
            alphabet[:self.axis],
            alphabet[self.axis:rank],
        )

        self.einsum_string_cols = '{}y{},z->{}{}zy'.format(
            alphabet[:self.axis],
            alphabet[self.axis:rank],
            alphabet[:self.axis],
            alphabet[self.axis:rank],
        )

        # self.rows1 = np.einsum(
        #     einsum_string_rows,
        #     indices,
        #     ones,
        # ).flatten()

        # self.cols1 = np.einsum(
        #     einsum_string_cols,
        #     indices,
        #     ones,
        # ).flatten()

        # # self.declare_partials(out_name, in1_name, rows=rows, cols=cols)

        # self.rows2 = np.einsum(
        #     einsum_string_rows,
        #     indices,
        #     ones,
        # ).flatten()

        # self.cols2 = np.einsum(
        #     einsum_string_cols,
        #     indices,
        #     ones,
        # ).flatten()

        # print(self.rows1.nbytes, self.rows2.nbytes, self.cols1.nbytes, self.cols2.nbytes)

        # self.declare_partials(out_name, in2_name, rows=rows, cols=cols)

    def get_evaluation(self, eval_block, vars):

        out = self.get_output_id(self.out_name)
        in1 = self.get_input_id(self.in1_name)
        in2 = self.get_input_id(self.in2_name)
        eval_block.write(f'{out} = np.cross({in1}, {in2}, axisa = {self.axis}, axisb = {self.axis}, axisc = {self.axis})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac, lazy):
        
        if not lazy:
            self.indices = get_array_indices(*self.shape)
            self.rows = np.einsum(
                self.einsum_string_rows,
                self.indices,
                self.ones,
            ).flatten()

            self.cols = np.einsum(
                self.einsum_string_cols,
                self.indices,
                self.ones,
            ).flatten()

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
                    
                    if not lazy:
                        if is_sparse_jac:
                            jac = sp.csc_matrix((val, (self.rows, self.cols)), shape=(self.outsize, self.insize))
                        else:
                            jac = np.zeros((self.outsize, self.insize))
                            jac[self.rows, self.cols] = val

                    else:
                        indices = get_array_indices(*self.shape)
                        rows = np.einsum(
                            self.einsum_string_rows,
                            indices,
                            self.ones,
                        ).flatten()

                        cols = np.einsum(
                            self.einsum_string_cols,
                            indices,
                            self.ones,
                        ).flatten()

                        if is_sparse_jac:
                            jac = sp.csc_matrix((val, (rows, cols)), shape=(self.outsize, self.insize))
                        else:
                            jac = np.zeros((self.outsize, self.insize))
                            jac[rows, cols] = val

                        del indices
                        del rows
                        del cols
                        del val
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

                    if not lazy:
                        if is_sparse_jac:
                            jac = sp.csc_matrix((val, (self.rows, self.cols)), shape=(self.outsize, self.insize))
                        else:
                            jac = np.zeros((self.outsize, self.insize))
                            jac[self.rows, self.cols] = val
                    else:
                        indices = get_array_indices(*self.shape)

                        rows = np.einsum(
                            self.einsum_string_rows,
                            indices,
                            self.ones,
                        ).flatten()

                        cols = np.einsum(
                            self.einsum_string_cols,
                            indices,
                            self.ones,
                        ).flatten()

                        if is_sparse_jac:
                            jac = sp.csc_matrix((val, (rows, cols)), shape=(self.outsize, self.insize))
                        else:
                            jac = np.zeros((self.outsize, self.insize))
                            jac[rows, cols] = val
                        del indices
                        del rows
                        del cols
                        del val

                    return jac

            func_name = self.operation.name+'_'+input+'_jac'
            vars[func_name] = compute_cross_jac

            # partials_block.write(f'import time as time\ns = time.time()')
            partials_block.write(f'{partial_name} = {func_name}({self.get_input_id(self.in1_name)}, {self.get_input_id(self.in2_name)})')
            # partials_block.write(f'print(time.time()-s, \'{partial_name}\', {self.outsize}, {self.insize})')

    def determine_sparse(self):
        # in1_sparsity = len(self.rows1)/(self.outsize*self.insize)
        # in2_sparsity = len(self.rows2)/(self.outsize*self.insize)

        if (self.outsize > 100) and (self.insize > 100):
            # print(in1_sparsity, in2_sparsity, self.outsize, self.insize)
            return True
        return False


    def get_accumulation_function(self, input_paths, path_output, partials_block, vars):
        # Here we generate code to continue jacobian accumulation given accumulated paths from output to this implicit operation

        # implicit solver object
        name = self.operation.name+'_vjp'
        vars[name] = self.accumulate_cross_rev

        # Ouput paths must be in correct order...
        in_argument = ''
        for inv in path_output:
            in_argument += inv+', '
        in_argument = in_argument.rstrip(in_argument[-1])
        in_argument = in_argument.rstrip(in_argument[-1])

        # give paths of outputs to inputs
        partials_block.write(f'{self.operation.name}_path_in = {name}({self.get_input_id(self.in1_name)},{self.get_input_id(self.in2_name)},{in_argument})')

        # Input paths must be in correct order...
        for i, path in enumerate(input_paths):
            partials_block.write(f'{path} = {self.operation.name}_path_in[{i}]')

    def is_jac_function(self, vjp=False):
        return vjp


    def compute_vjp_a_b(self, a, b, vjp_z):
        vjp_z = vjp_z.reshape(self.shape)

        # Compute the VJP for vector 'a'
        vjp_a = -np.cross(vjp_z, b, axisa = self.axis, axisb = self.axis, axisc =self.axis)
        
        # Compute the VJP for vector 'b'
        vjp_b = -np.cross(a, vjp_z, axisa = self.axis, axisb = self.axis, axisc = self.axis)
        
        return vjp_a, vjp_b

    def accumulate_cross_rev(self, a, b, vjp_z_mat):

        vjp_a_mat = np.zeros((vjp_z_mat.shape[0], a.size))
        vjp_b_mat = np.zeros((vjp_z_mat.shape[0], b.size))

        to_array = False
        if sp.issparse(vjp_z_mat):
            to_array = True
        for i, row in enumerate(range(vjp_z_mat.shape[0])):
            if to_array:
                vjp_z = vjp_z_mat[row,:].toarray()
            else:
                vjp_z = vjp_z_mat[row,:]
            vjp_a, vjp_b = self.compute_vjp_a_b(a, b, vjp_z)
            vjp_a_mat[row,:] = vjp_a.flatten()
            vjp_b_mat[row,:] = vjp_b.flatten()
        return vjp_a_mat, vjp_b_mat


def get_array_indices(*shape):
    return np.arange(np.prod(shape)).reshape(shape)


