from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_list, get_scalars_list
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.sparse_utils import sparse_matrix

import scipy.sparse as sp
import numpy as np


def get_pnorm_lite(op):

    if op.literals['axis'] == None:
        return VectorizedPNormLite
    else:
        return VectorizedAxisWisePNormLite


class VectorizedPNormLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'pnorm'
        name = f'{name} {op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.shape = operation.dependencies[0].shape
        self.in_size = np.prod(operation.dependencies[0].shape)
        self.in_name = operation.dependencies[0].name
        self.out_name = operation.outs[0].name
        self.pnorm_type = operation.literals['pnorm_type']
        # self.val = operation.dependencies[0].val

    def get_evaluation(self, eval_block, vars):

        eval_block.write(f'{self.get_output_id(self.out_name)} = np.linalg.norm({self.get_input_id(self.in_name)}.flatten(), ord={self.pnorm_type})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):
        key_tuple = get_only(partials_dict)
        input = key_tuple[1].id
        output = key_tuple[0].id
        partial_name = partials_dict[key_tuple]['name']
        partials_block.write(f'{output} = np.linalg.norm({input}.flatten(), ord={self.pnorm_type})')
        if is_sparse_jac:
            partials_block.write(f'{partial_name} = sp.csc_matrix(np.array([{output}**({1-self.pnorm_type})*({input}**({self.pnorm_type-1})).flatten()]))')
        else:
            partials_block.write(f'{partial_name} = np.array([{output}**({1-self.pnorm_type})*({input}**({self.pnorm_type-1})).flatten()])')

    def determine_sparse(self):
        return False


class VectorizedAxisWisePNormLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'pnorm_axis'
        name = f'{name} {op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.shape = operation.dependencies[0].shape
        self.in_size = np.prod(operation.dependencies[0].shape)
        self.in_name = operation.dependencies[0].name
        self.out_name = operation.outs[0].name
        self.pnorm_type = operation.literals['pnorm_type']
        # self.val = operation.dependencies[0].val
        self.axis = operation.literals['axis']
        self.out_shape = tuple(np.delete(operation.dependencies[0].shape, operation.literals['axis']))

        # Computation of the einsum string that will be used in partials
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        rank = len(self.shape)
        input_subscripts = alphabet[:rank]
        output_subscripts = np.delete(list(input_subscripts), self.axis)
        output_subscripts = ''.join(output_subscripts)

        self.operation = '{},{}->{}'.format(
            output_subscripts,
            input_subscripts,
            input_subscripts,
        )

        # Computation of Output shape if the shape is not provided
        if self.out_shape == None:
            output_shape = np.delete(self.shape, axis)
            self.output_shape = tuple(output_shape)
        else:
            self.output_shape = self.out_shape
        self.out_size = np.prod(self.out_shape)

        # Defining the rows and columns of the sparse partial matrix
        input_size = np.prod(self.shape)
        self.cols = np.arange(input_size)
        rows = np.unravel_index(np.arange(input_size), shape=self.shape)
        rows = np.delete(np.array(rows), self.axis, axis=0)
        self.rows = np.ravel_multi_index(rows, dims=self.output_shape)

    def get_evaluation(self, eval_block, vars):

        eval_block.write(f'{self.get_output_id(self.out_name)} = np.sum({self.get_input_id(self.in_name)}**{self.pnorm_type},axis={self.axis})**(1 / {self.pnorm_type})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):

        key_tuple = get_only(partials_dict)
        input = key_tuple[1].id
        output = key_tuple[0].id
        partial_name = partials_dict[key_tuple]['name']

        row_name = partial_name+'_rows'
        col_name = partial_name+'_cols'
        vars[row_name] = self.rows
        vars[col_name] = self.cols

        partials_block.write(f'{output} = np.sum({input}**{self.pnorm_type},axis={self.axis})**(1 / {self.pnorm_type})')
        if is_sparse_jac:
            # csc_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
            partials_block.write(f'temp = np.einsum(\'{self.operation}\', {output}**(1 - {self.pnorm_type}), {input}**({self.pnorm_type} - 1)).flatten()')
            partials_block.write(f'{partial_name} = sp.csc_matrix((temp,({row_name},{col_name})), shape = ({self.out_size}, {self.in_size}))')
        else:
            vars[partial_name] = np.zeros((self.out_size, self.in_size))
            partials_block.write(f'{partial_name}[{row_name},{col_name}] = np.einsum(\'{self.operation}\', {output}**(1 - {self.pnorm_type}), {input}**({self.pnorm_type} - 1)).flatten()')

    def determine_sparse(self):
        return False
