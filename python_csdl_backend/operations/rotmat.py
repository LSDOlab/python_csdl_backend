from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_list, get_scalars_list
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
import numpy as np
import scipy.sparse as sp


def get_rotmat_lite(op):
    return RotmatLite


class RotmatLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'rotmat'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        shape = operation.dependencies[0].shape
        in_name = operation.dependencies[0].name
        out_name = operation.outs[0].name
        self.out_name = out_name
        axis = operation.literals['axis']
        val = operation.dependencies[0].val

        if shape == (1, ):
            output_shape = (3, 3)
        else:
            output_shape = shape + (3, 3)

        self.output_shape = output_shape
        self.in_size = np.prod(shape)

        self.input_name = self.get_input_id(in_name)
        self.output_name = self.get_output_id(self.out_name)

        self.rows = np.arange(np.prod(output_shape))
        self.cols = np.einsum('...,ij->...ij', np.arange(np.prod(shape)),
                              np.ones((3, 3), int)).flatten()

        if axis == 'x':
            self.i_cos1, self.j_cos1 = 1, 1
            self.i_cos2, self.j_cos2 = 2, 2
            self.i_sin1, self.j_sin1 = 1, 2
            self.i_sin2, self.j_sin2 = 2, 1
            self.i_one, self.j_one = 0, 0
        elif axis == 'y':
            self.i_cos1, self.j_cos1 = 0, 0
            self.i_cos2, self.j_cos2 = 2, 2
            self.i_sin1, self.j_sin1 = 2, 0
            self.i_sin2, self.j_sin2 = 0, 2
            self.i_one, self.j_one = 1, 1
        elif axis == 'z':
            self.i_cos1, self.j_cos1 = 0, 0
            self.i_cos2, self.j_cos2 = 1, 1
            self.i_sin1, self.j_sin1 = 0, 1
            self.i_sin2, self.j_sin2 = 1, 0
            self.i_one, self.j_one = 2, 2

    def get_evaluation(self, eval_block, vars):

        def compute_rotmat(input_val):

            out_val = np.zeros(self.output_shape)
            out_val[..., self.i_cos1, self.j_cos1] = np.cos(input_val)
            out_val[..., self.i_cos2, self.j_cos2] = np.cos(input_val)
            out_val[..., self.i_sin1, self.j_sin1] = -np.sin(input_val)
            out_val[..., self.i_sin2, self.j_sin2] = np.sin(input_val)
            out_val[..., self.i_one, self.i_one] = 1.

            return out_val

        func_name = f'{self.name}_func'
        vars[func_name] = compute_rotmat
        eval_block.write(f'{self.output_name} = {func_name}({self.input_name})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):

        key_tuple = get_only(partials_dict)
        input = key_tuple[1].id
        output = key_tuple[0].id
        partial_name = partials_dict[key_tuple]['name']
        self.out_size = np.prod(self.output_shape)

        def deriv_rotmat(input_val):
            out_shape = self.operation.dependencies[0].shape + (3, 3)
            a = input_val

            derivs = np.zeros(out_shape)
            print(derivs.shape)
            derivs[..., self.i_cos1, self.j_cos1] = -np.sin(a)
            derivs[..., self.i_cos2, self.j_cos2] = -np.sin(a)
            derivs[..., self.i_sin1, self.j_sin1] = -np.cos(a)
            derivs[..., self.i_sin2, self.j_sin2] = np.cos(a)
            derivs = derivs.flatten()

            if is_sparse_jac:
                return sp.csc_matrix((derivs, (self.rows, self.cols)), shape=(self.out_size, self.in_size))
            else:
                totals = np.zeros((self.out_size, self.in_size))
                totals[self.rows, self.cols] = derivs
                return totals

        partial_f_name = f'{self.name}_{self.output_name}'
        vars[partial_f_name] = deriv_rotmat
        partials_block.write(f'{partial_name} = {partial_f_name}({self.input_name})')
