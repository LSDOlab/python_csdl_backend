from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_list, get_scalars_list
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
from python_csdl_backend.utils.sparse_utils import get_sparsity
from python_csdl_backend.operations.max import ScalarExtremumLite
import numpy as np
import scipy.sparse as sp


def get_average_lite(op):
    raise NotImplementedError('average not yet implemented')
    if len(op.dependencies) == 1 and op.literals['axes'] != None:
        return SingleTensorAverageLite
    elif len(op.dependencies) > 1 and op.literals['axes'] == None:
        return ElementwiseAverageLite
    elif len(op.dependencies) == 1 and op.literals['axes'] == None:
        return SingleTensorAverageLite


# class AxisAverageLite(OperationBase):

#     def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
#         op_name = 'axisaverage'
#         name = f'{name}_{op_name}'
#         super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

#         shape = operation.dependencies[0].shape
#         in_name = operation.dependencies[0].name
#         self.in_id = self.get_input_id(in_name)
#         axis = operation.literals['axis']
#         out_name = operation.outs[0].name
#         self.out_id = self.get_output_id(out_name)
#         self.rho = operation.literals['rho']
#         self.val = operation.dependencies[0].val

#         total_rank = len(shape)
#         if axis < 0:
#             axis += total_rank
#         self.axis = axis

#         in_shape = tuple(shape)
#         out_shape = shape[:axis] + shape[axis + 1:]

#         self.outsize = np.prod(out_shape)
#         self.insize = np.prod(in_shape)

#         out_indices = np.arange(np.prod(out_shape)).reshape(out_shape)
#         in_indices = np.arange(np.prod(in_shape)).reshape(in_shape)

#         alphabet = 'abcdefghijkl'

#         self.einsum_str = einsum_str = '{},{}->{}'.format(
#             alphabet[:axis] + alphabet[axis + 1:total_rank],
#             alphabet[axis],
#             alphabet[:total_rank],
#         )
#         self.ones = ones = np.ones(shape[axis])

#         self.rows = np.einsum(
#             einsum_str,
#             out_indices,
#             ones.astype(int),
#         ).flatten()
#         self.cols = in_indices.flatten()

#     def get_evaluation(self, eval_block, vars):

#         def compute_average(in_val_val):
#             in_val = -in_val_val
#             g_max = np.max(in_val, axis=self.axis)
#             g_diff = in_val - np.einsum(
#                 self.einsum_str,
#                 g_max,
#                 self.ones,
#             )
#             exponents = np.exp(self.rho * g_diff)
#             summation = np.sum(exponents, axis=self.axis)
#             result = -g_max - 1.0 / self.rho * np.log(summation)
#             return result

#         vars[self.name] = compute_average
#         eval_block.write(f'{self.out_id} = {self.name}({self.in_id})')

#     def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):

#         key_tuple = get_only(partials_dict)
#         input = key_tuple[1].id
#         output = key_tuple[0].id
#         partial_name = partials_dict[key_tuple]['name']

#         def compute_average_deriv(in_val_val):
#             in_val = - in_val_val
#             g_max = np.max(in_val, axis=self.axis)
#             g_diff = in_val - np.einsum(
#                 self.einsum_str,
#                 g_max,
#                 self.ones,
#             )
#             exponents = np.exp(self.rho * g_diff)
#             summation = np.sum(exponents, axis=self.axis)

#             dsum_dg = self.rho * exponents
#             dKS_dsum = 1.0 / (self.rho * np.einsum(
#                 self.einsum_str,
#                 summation,
#                 self.ones,
#             ))
#             dKS_dg = dKS_dsum * dsum_dg
#             return dKS_dg

#         partials_name = self.name+'_partials'
#         row_name = self.name+'_rows'
#         col_name = self.name+'_cols'

#         vars[partials_name] = compute_average_deriv
#         vars[row_name] = self.rows
#         vars[col_name] = self.cols

#         partials_block.write(f'temp = {partials_name}({input}).flatten()')
#         if is_sparse_jac:
#             partials_block.write(f'{partial_name} = sp.csc_matrix((temp,({row_name},{col_name})),shape = ({self.outsize},{self.insize}))')
#         else:
#             partials_block.write(f'{partial_name} = np.zeros(({self.outsize},{self.insize}))')
#             partials_block.write(f'{partial_name}[{row_name},{col_name}] = temp')

#     def deteraveragee_sparse(self):
#         if self.insize < 100:
#             return False

#         if get_sparsity(len(self.rows), self.outsize, self.insize) < SPARSE_SIZE_CUTOFF:
#             return True
#         return False


# class ElementwiseAverageLite(OperationBase):

#     def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
#         op_name = 'elementaverage'
#         name = f'{name}_{op_name}'
#         super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

#         self.shape = operation.dependencies[0].shape
#         self.in_names = [var.name for var in operation.dependencies]
#         self.in_ids = [self.get_input_id(in_name) for in_name in self.in_names]
#         self.out_name = operation.outs[0].name
#         self.out_id = self.get_output_id(self.out_name)
#         self.rho = operation.literals['rho']
#         self.vals = [var.val for var in operation.dependencies]

#         r_c = np.arange(np.prod(self.shape))
#         self.rows = self.cols = r_c

#         self.outsize = np.prod(self.shape)
#         self.insize = np.prod(self.shape)

#     def get_evaluation(self, eval_block, vars):

#         eval_block.write(f'fmax = -{self.in_ids[0]} - 1')
#         for in_name in self.in_ids:
#             eval_block.write(f'fmax = np.maximum(fmax, -{in_name})')
#         eval_block.write(f'arg = 0.0')
#         for in_name in self.in_ids:
#             eval_block.write(f'arg += np.exp({self.rho} * (-{in_name} - fmax))')
#         eval_block.write(f'{self.out_id} = -(fmax + 1. / {self.rho} * np.log(arg))')

#     def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):

#         row_name = self.name+'_rows'
#         col_name = self.name+'_cols'
#         vars[row_name] = self.rows
#         vars[col_name] = self.cols

#         partials_block.write(f'fmax = -{self.in_ids[0]} - 1')
#         for in_name in self.in_ids:
#             partials_block.write(f'fmax = np.maximum(fmax, -{in_name})')
#         partials_block.write(f'arg = 0.0')
#         for in_name in self.in_ids:
#             partials_block.write(f'arg += np.exp({self.rho} * (-{in_name} - fmax))')

#         for key_tuple in partials_dict:
#             input = key_tuple[1].id
#             output = key_tuple[0].id
#             partial_name = partials_dict[key_tuple]['name']

#             partials_block.write(f'temp = (1. / arg * np.exp({self.rho} * (-{input} - fmax))).flatten()')
#             if is_sparse_jac:
#                 partials_block.write(f'{partial_name} = sp.csc_matrix((temp,({row_name},{col_name})),shape = ({self.outsize},{self.insize}))')
#             else:
#                 partials_block.write(f'{partial_name} = np.zeros(({self.outsize},{self.insize}))')
#                 partials_block.write(f'{partial_name}[{row_name},{col_name}] = temp')

#     def deteraveragee_sparse(self):
#         if self.insize < 100:
#             return False

#         if get_sparsity(len(self.rows), self.outsize, self.insize) < SPARSE_SIZE_CUTOFF:
#             return True
#         return False


class SingleTensorAverageLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        name = 'average'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.in_name = operation.dependencies[0].name
        self.shape = operation.dependencies[0].shape
        self.out_name = operation.outs[0].name
        self.out_shape = operation.outs[0].shape
        self.axes = operation.literals['axes']
        # self.val = operation.dependencies[0].val

        in_name = self.in_name
        shape = self.shape
        out_name = self.out_name
        out_shape = self.out_shape
        print(out_shape)
        axes = self.axes
        # val = self.val

        # Computation of Output shape if the shape is not provided
        if out_shape != None:
            self.output_shape = out_shape
        elif axes != None:
            output_shape = np.delete(shape, axes)
            self.output_shape = tuple(output_shape)

        input_size = np.prod(shape)
        self.input_size = input_size
        self.output_size = np.prod(self.output_shape)

        # axes == None works for any tensor
        # self.add_input(in_name, shape=shape, val=val)
        if axes == None:
            self.val = np.full((input_size, ), 1. / input_size)
            self.nnz = len(self.val.flatten())

        # axes != None works only for matrices
        else:
            # self.add_output(out_name, shape=self.output_shape)
            self.cols = np.arange(input_size)

            rows = np.unravel_index(np.arange(input_size), shape=shape)
            rows = np.delete(np.array(rows), axes, axis=0)
            self.rows = np.ravel_multi_index(rows, dims=self.output_shape)

            num_entries_averaged = np.prod(np.array(shape)[axes])
            self.val = np.full((input_size, ), 1. / num_entries_averaged)
            self.nnz = len(self.val.flatten())

    def get_evaluation(self, eval_block, vars):
        in_name = self.in_name
        out_name = self.out_name
        axes = self.axes

        # axes == None works for any tensor
        if axes == None:
            eval_block.write(f'{self.get_output_id(out_name)} = np.array(np.average({self.get_input_id(in_name)})).reshape((1,))')
            # eval_block.write(f'{self.get_output_id(out_name)} = np.average({self.get_input_id(in_name)})')

        # axes != None works only for matrices
        else:
            axis_name = f'{self.get_input_id(in_name)}_{self.get_output_id(out_name)}_axes'
            vars[axis_name] = axes
            eval_block.write(f'{self.get_output_id(out_name)} = np.average({self.get_input_id(in_name)}, axis={axis_name})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):

        key_tuple = get_only(partials_dict)
        input = key_tuple[1].id
        output = key_tuple[0].id
        partial_name = partials_dict[key_tuple]['name']

        if self.axes == None:
            if is_sparse_jac:
                vars[partial_name] = sp.csc_matrix(self.val)
            else:
                vars[partial_name] = self.val.reshape((self.input_size))
        else:
            if is_sparse_jac:
                vars[partial_name] = sp.csc_matrix((self.val.flatten(), (self.rows, self.cols)), shape=(self.output_size, self.input_size))
            else:
                vars[partial_name] = np.zeros((self.output_size, self.input_size))
                vars[partial_name][self.rows, self.cols] = self.val.flatten()

    def determine_sparse(self):
        if self.input_size < 100:
            return False

        if get_sparsity(self.nnz, self.output_size, self.input_size) < SPARSE_SIZE_CUTOFF:
            return True
        return False
