from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_unique_list, get_scalars_list
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
import scipy.sparse as sp
import numpy as np


def get_linear_combination_lite(op):
    return LinearCombinationLite


class LinearCombinationLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'linear_combination'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.constant = self.operation.literals['constant']
        self.in_names = [d.name for d in self.operation.dependencies]
        self.coeffs = get_scalars_list(self.operation.literals['coeffs'], self.in_names)
        self.input_size = np.prod(self.operation.dependencies[0].shape)

        self.out_name = self.operation.outs[0].name
        self.shape = self.operation.outs[0].shape

    def get_evaluation(self, eval_block, vars):

        # Initialize constant
        const_name = self.operation.name+'_constant'
        start_constant = True
        if isinstance(self.constant, np.ndarray):
            vars[const_name] = self.constant.reshape(self.shape)
        else:
            if self.constant != 0:
                vars[const_name] = self.constant
            else:
                start_constant = False

        if start_constant:
            eval_block.write(f'{self.get_output_id(self.out_name)} = {const_name}')
        else:
            eval_block.write(f'{self.get_output_id(self.out_name)} = ')

        # write linear combination
        num_combination = 0
        for in_name, coeff in zip(self.in_names, self.coeffs):
            
            if coeff != 1:
                coeff_str = f'{coeff}*'
            else:
                coeff_str = ''

            if num_combination == 0 and not start_constant:
                starting_str = ''
            else:
                starting_str = '+'
            eval_block.write(f'{starting_str}{coeff_str}{self.get_input_id(in_name)}', linebreak=False)

            num_combination += 1
    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac, lazy):

        for key_tuple in partials_dict:
            input = key_tuple[1].id
            output = key_tuple[0].id
            partial_name = partials_dict[key_tuple]['name']
            lang_input = self.get_lang_input(input)

            coeff = self.coeffs[self.in_names.index(lang_input)]
            size = self.input_size

            # OLD FULL JACOBIAN
            # if is_sparse_jac:
            #     vars[partial_name] = sp.eye(size, format='csc')*coeff
            # else:
            #     vars[partial_name] = np.eye(size)*coeff

            # NEW:
            # only return diag values for elementwise
            # Also sparsity doesn't matter

            if lazy:
                partials_block.write(f'{partial_name} = np.ones({size})*{coeff}')        
            else:
                vars[partial_name] = np.ones(size)*coeff

    def determine_sparse(self):
        return self.determine_sparse_default_elementwise(self.input_size)
