from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_list, get_scalars_list
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.sparse_utils import sparse_matrix

import numpy as np
import scipy.sparse as sp


def get_expand_lite(op):

    if (op.dependencies[0].shape != (1, )):
    # if len(op.dependencies[0].shape) != 1:
        return ExpandArrayLite
    else:
        return ExpandScalarLite


class ExpandArrayLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'expand_array'
        name = f'{name} {op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        input_name_id = get_only(self.nx_inputs_dict)
        output_name_id = get_only(self.nx_outputs_dict)

        self.inname = input_name_id
        self.outname = output_name_id

        self.outvar = self.nx_outputs_dict[output_name_id].var
        self.invar = self.nx_inputs_dict[input_name_id].var

        self.out_shape = self.outvar.shape
        self.val = self.invar.val
        self.expand_indices = operation.literals['expand_indices']
        print(self.expand_indices)

        # self.outname = get_only(self.nx_outputs_dict)
        # self.outvar = self.nx_outputs_dict[self.outname]

        # self.inname = get_only(self.nx_inputs_dict)
        # self.invar = self.nx_inputs_dict[self.inname]

        # self.out_shape = self.outvar.shape
        # self.expand_indices = operation.literals['expand_indices']
        # self.val = self.invar.val

        (
            in_string,
            ones_string,
            out_string,
            self.in_shape,
            ones_shape,
        ) = decompose_shape_tuple(self.out_shape, self.expand_indices)

        einsum_string = '{},{}->{}'.format(in_string, ones_string, out_string)

        in_indices = get_array_indices(*self.in_shape)
        out_indices = get_array_indices(*self.out_shape)

        self.einsum_string = einsum_string
        self.ones_shape = ones_shape

        self.rows = out_indices.flatten()
        self.cols = np.einsum(einsum_string, in_indices, np.ones(ones_shape,
                                                                 int)).flatten()
        # self.declare_partials(out_name, self.inname, val=1., rows=rows, cols=cols)

    def get_evaluation(self, eval_block, vars):

        # eval_block.write(f'print({self.inname},{self.inname}.shape )')
        eval_block.write(f'{self.outname} = np.einsum(\'{self.einsum_string}\', {self.inname}.reshape({self.in_shape}) ,np.ones({self.ones_shape})).reshape({self.out_shape})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):

        key_tuple = get_only(partials_dict)
        input = key_tuple[1].id
        output = key_tuple[0].id
        partial_name = partials_dict[key_tuple]['name']

        size = np.prod(self.invar.shape)
        sizeout = np.prod(self.outvar.shape)

        val = np.ones(len(self.rows))

        if size < SPARSE_SIZE_CUTOFF:
            vars[partial_name] = sp.csc_matrix((val, (self.rows, self.cols)), shape=(sizeout, size)).toarray()
        else:
            vars[partial_name] = sp.csc_matrix((val, (self.rows, self.cols)), shape=(sizeout, size))

    def determine_sparse(self):

        size = np.prod(self.invar.shape)
        sizeout = np.prod(self.outvar.shape)
        if (size*sizeout) < 10000:
            return False

        if len(self.rows)/(size*sizeout) < 0.66:
            return True
        else:
            return False



class ExpandScalarLite(OperationBase):
    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'expand_scalar'
        name = f'{name} {op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        input_name_id = get_only(self.nx_inputs_dict)
        output_name_id = get_only(self.nx_outputs_dict)

        self.inname = input_name_id
        self.outname = output_name_id

        self.outvar = self.nx_outputs_dict[output_name_id].var
        self.invar = self.nx_inputs_dict[input_name_id].var

        self.out_shape = self.outvar.shape
        self.val = self.invar.val

    def get_evaluation(self, eval_block, vars):

        eval_block.write(f'{self.outname} = np.empty({self.out_shape})')
        eval_block.write(f'{self.outname}.fill({self.inname}.item())')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):

        key_tuple = get_only(partials_dict)
        input = key_tuple[1].id
        output = key_tuple[0].id
        partial_name = partials_dict[key_tuple]['name']

        rows = np.arange(np.prod(self.out_shape))
        cols = np.zeros(np.prod(self.out_shape), int)
        data = np.ones(np.prod(self.out_shape))

        size = np.prod(self.invar.shape)
        sizeout = np.prod(self.out_shape)

        if not is_sparse_jac:
            vars[partial_name] = sp.csc_matrix((data, (rows, cols))).toarray()
        else:
            vars[partial_name] = sp.csc_matrix((data, (rows, cols)))


def decompose_shape_tuple(shape, select_indices):
    alphabet = 'abcdefghij'

    einsum_selection = ''
    einsum_full = ''
    einsum_remainder = ''
    shape_selection = []
    shape_remainder = []
    for index in range(len(shape)):
        if index not in select_indices:
            einsum_selection += alphabet[index]
            shape_selection.append(shape[index])
        else:
            einsum_remainder += alphabet[index]
            shape_remainder.append(shape[index])
        einsum_full += alphabet[index]

    shape_selection = tuple(shape_selection)
    shape_remainder = tuple(shape_remainder)

    return (
        einsum_selection,
        einsum_remainder,
        einsum_full,
        shape_selection,
        shape_remainder,
    )


def get_array_indices(*shape):
    return np.arange(np.prod(shape)).reshape(shape)
