from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_unique_list, get_scalars_list
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
from python_csdl_backend.utils.sparse_utils import get_sparsity
import numpy as np
import scipy.sparse as sp


def get_max_lite(op):
    if len(op.dependencies) == 1 and op.literals['axis'] != None:
        return AxisMaxLite
    elif len(op.dependencies) > 1 and op.literals['axis'] == None:
        return ElementwiseMaxLite
    elif len(op.dependencies) == 1 and op.literals['axis'] == None:
        return ScalarExtremumMaxLite


class AxisMaxLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'axismax'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        shape = operation.dependencies[0].shape
        in_name = operation.dependencies[0].name
        self.in_id = self.get_input_id(in_name)
        axis = operation.literals['axis']
        out_name = operation.outs[0].name
        self.out_id = self.get_output_id(out_name)
        self.rho = operation.literals['rho']
        # self.val = operation.dependencies[0].val

        total_rank = len(shape)
        if axis < 0:
            axis += total_rank
        self.axis = axis

        in_shape = tuple(shape)
        out_shape = shape[:axis] + shape[axis + 1:]

        self.outsize = np.prod(out_shape)
        self.insize = np.prod(in_shape)

        out_indices = np.arange(np.prod(out_shape)).reshape(out_shape)
        in_indices = np.arange(np.prod(in_shape)).reshape(in_shape)

        alphabet = 'abcdefghijkl'

        self.einsum_str = einsum_str = '{},{}->{}'.format(
            alphabet[:axis] + alphabet[axis + 1:total_rank],
            alphabet[axis],
            alphabet[:total_rank],
        )
        self.ones = ones = np.ones(shape[axis])

        self.rows = np.einsum(
            einsum_str,
            out_indices,
            ones.astype(int),
        ).flatten()
        self.cols = in_indices.flatten()

    def get_evaluation(self, eval_block, vars):

        def compute_max(in_val):
            g_max = np.max(in_val, axis=self.axis)
            g_diff = in_val - np.einsum(
                self.einsum_str,
                g_max,
                self.ones,
            )
            exponents = np.exp(self.rho * g_diff)
            summation = np.sum(exponents, axis=self.axis)
            result = g_max + 1.0 / self.rho * np.log(summation)
            return result

        vars[self.name] = compute_max
        eval_block.write(f'{self.out_id} = {self.name}({self.in_id})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac, lazy):

        key_tuple = get_only(partials_dict)
        input = key_tuple[1].id
        output = key_tuple[0].id
        partial_name = partials_dict[key_tuple]['name']

        def compute_max_deriv(in_val):
            g_max = np.max(in_val, axis=self.axis)
            g_diff = in_val - np.einsum(
                self.einsum_str,
                g_max,
                self.ones,
            )
            exponents = np.exp(self.rho * g_diff)
            summation = np.sum(exponents, axis=self.axis)

            dsum_dg = self.rho * exponents
            dKS_dsum = 1.0 / (self.rho * np.einsum(
                self.einsum_str,
                summation,
                self.ones,
            ))
            dKS_dg = dKS_dsum * dsum_dg
            return dKS_dg

        partials_name = self.name+'_partials'
        row_name = self.name+'_rows'
        col_name = self.name+'_cols'

        vars[partials_name] = compute_max_deriv
        vars[row_name] = self.rows
        vars[col_name] = self.cols

        partials_block.write(f'temp = {partials_name}({input}).flatten()')
        if is_sparse_jac:
            partials_block.write(f'{partial_name} = sp.csc_matrix((temp,({row_name},{col_name})),shape = ({self.outsize},{self.insize}))')
        else:
            partials_block.write(f'{partial_name} = np.zeros(({self.outsize},{self.insize}))')
            partials_block.write(f'{partial_name}[{row_name},{col_name}] = temp')

    def determine_sparse(self):
        if self.insize < 100:
            return False

        if get_sparsity(len(self.rows), self.outsize, self.insize) < SPARSE_SIZE_CUTOFF:
            return True
        return False


class ElementwiseMaxLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'elementmax'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.shape = operation.dependencies[0].shape
        self.in_names = [var.name for var in operation.dependencies]
        self.in_ids = [self.get_input_id(in_name) for in_name in self.in_names]
        self.out_name = operation.outs[0].name
        self.out_id = self.get_output_id(self.out_name)
        self.rho = operation.literals['rho']
        # self.vals = [var.val for var in operation.dependencies]

        r_c = np.arange(np.prod(self.shape))
        self.rows = self.cols = r_c

        self.outsize = np.prod(self.shape)
        self.insize = np.prod(self.shape)

    def get_evaluation(self, eval_block, vars):

        eval_block.write(f'fmax = {self.in_ids[0]} - 1')
        for in_name in self.in_ids:
            eval_block.write(f'fmax = np.maximum(fmax, {in_name})')
        eval_block.write(f'arg = 0.0')
        for in_name in self.in_ids:
            eval_block.write(f'arg += np.exp({self.rho} * ({in_name} - fmax))')
        eval_block.write(f'{self.out_id} = (fmax + 1. / {self.rho} * np.log(arg))')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac, lazy):

        row_name = self.name+'_rows'
        col_name = self.name+'_cols'
        vars[row_name] = self.rows
        vars[col_name] = self.cols

        partials_block.write(f'fmax = {self.in_ids[0]} - 1')
        for in_name in self.in_ids:
            partials_block.write(f'fmax = np.maximum(fmax, {in_name})')
        partials_block.write(f'arg = 0.0')
        for in_name in self.in_ids:
            partials_block.write(f'arg += np.exp({self.rho} * ({in_name} - fmax))')

        for key_tuple in partials_dict:
            input = key_tuple[1].id
            output = key_tuple[0].id
            partial_name = partials_dict[key_tuple]['name']

            partials_block.write(f'temp = (1. / arg * np.exp({self.rho} * ({input} - fmax))).flatten()')
            if is_sparse_jac:
                partials_block.write(f'{partial_name} = sp.csc_matrix((temp,({row_name},{col_name})),shape = ({self.outsize},{self.insize}))')
            else:
                partials_block.write(f'{partial_name} = np.zeros(({self.outsize},{self.insize}))')
                partials_block.write(f'{partial_name}[{row_name},{col_name}] = temp')

    def determine_sparse(self):
        if self.insize < 100:
            return False

        if get_sparsity(len(self.rows), self.outsize, self.insize) < SPARSE_SIZE_CUTOFF:
            return True
        return False


class ScalarExtremumLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'scalar_extremum_max'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        shape = operation.dependencies[0].shape
        in_name = operation.dependencies[0].name
        self.in_id = self.get_input_id(in_name)
        out_name = operation.outs[0].name
        self.out_id = self.get_output_id(out_name)
        self.rho = operation.literals['rho']
        # self.val = operation.dependencies[0].val

        in_shape = tuple(shape)
        out_shape = (1,)
        self.shape = shape

        self.outsize = np.prod(out_shape)
        self.insize = np.prod(in_shape)

        # out_indices = np.arange(np.prod(out_shape)).reshape(out_shape)
        # in_indices = np.arange(np.prod(in_shape)).reshape(in_shape)

    def get_evaluation(self, eval_block, vars):

        def compute_max(in_val):
            if self.lower_flag:
                g_max = np.max(-in_val)
                g_diff = -in_val - g_max
            else:
                g_max = np.max(in_val)
                g_diff = in_val - g_max

            exponents = np.exp(self.rho * g_diff)
            summation = np.sum(exponents)
            result = (g_max + 1.0 / self.rho * np.log(summation)).reshape(1,)
            return result

        vars[self.name] = compute_max
        if self.lower_flag:
            eval_block.write(f'{self.out_id} = -{self.name}({self.in_id})')
        else:
            eval_block.write(f'{self.out_id} = {self.name}({self.in_id})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac, lazy):

        key_tuple = get_only(partials_dict)
        input = key_tuple[1].id
        output = key_tuple[0].id
        partial_name = partials_dict[key_tuple]['name']

        def compute_max_deriv(in_val):
            if self.lower_flag:
                g_max = np.max(-in_val)
                g_diff = -in_val - g_max
            else:
                g_max = np.max(in_val)
                g_diff = in_val - g_max

            exponents = np.exp(self.rho * g_diff)
            summation = np.sum(exponents)

            dsum_dg = self.rho * exponents
            dKS_dsum = 1.0 / (self.rho * summation * np.ones(self.shape))
            # print(dKS_dsum.shape, dsum_dg.shape)
            dKS_dg = (dKS_dsum * dsum_dg).reshape((self.outsize, self.insize))
            return dKS_dg

        partials_name = self.name+'_partials'+partial_name

        vars[partials_name] = compute_max_deriv

        partials_block.write(f'temp = {partials_name}({input})')
        if is_sparse_jac:
            partials_block.write(f'{partial_name} = sp.csc_matrix(temp)')
        else:
            partials_block.write(f'{partial_name}= temp')

    def determine_sparse(self):
        if self.insize < 100:
            return False
        return True

class ScalarExtremumMaxLite(ScalarExtremumLite):

    def __init__(self, operation, nx_inputs, nx_outputs,name='', **kwargs):
        name = 'max'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)
        self.lower_flag = False
