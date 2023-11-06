from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_unique_list, get_scalars_list
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF

import numpy as np


def get_eigenvalues_lite(op):
    return EigenvaluesLite


class EigenvaluesLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'eigenvalues'
        name = f'{name} {op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        input_name_id = get_only(self.nx_inputs_dict)
        self.output_name_id_real = self.get_output_id(self.operation.outs[0].name)
        self.output_name_id_imag = self.get_output_id(self.operation.outs[1].name)
        
        self.n = operation.literals['n']
        self.input_size = self.n**2
        self.input_name = input_name_id

    def get_evaluation(self, eval_block, vars):

        eval_block.write(f'w, _  = np.linalg.eig({self.input_name})')
        eval_block.write(f'idx = w.argsort()[::-1]')
        eval_block.write(f'w = w[idx]')
        eval_block.write(f'{self.output_name_id_real}  = np.real(w)')
        eval_block.write(f'{self.output_name_id_imag}  = np.imag(w)')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac, lazy):

        func_name = self.operation.name+'_jac'+self.output_name_id_real+self.output_name_id_imag

        def compute_jacs(input_matrix):
            size = self.n
            shape = (size, size)

            # v are the eigenvectors in each columns
            w, v = np.linalg.eig(input_matrix)
            idx = w.argsort()[::-1]   
            w = w[idx]
            v = v[:,idx]

            # v inverse transpose
            v_inv_T = (np.linalg.inv(v)).T

            # preallocate Jacobian: n outputs, n^2 inputs
            temp_r = np.zeros((size, size*size))
            temp_i = np.zeros((size, size*size))

            for j in range(len(w)):

                # dA/dw(j,:) = v(:,j)*(v^-T)(:j)
                partial = np.outer(v[:, j], v_inv_T[:, j]).flatten(order='F')
                # Note that the order of flattening matters, hence argument in flatten()

                # Set jacobian rows
                temp_r[j, :] = np.real(partial)
                temp_i[j, :] = np.imag(partial)

            # Set Jacobian
            return temp_r, temp_i
        vars[func_name] = compute_jacs
        partials_block.write(f'temp_r, temp_i = {func_name}({self.input_name})')

        for key_tuple in partials_dict:
            input = key_tuple[1].id
            output = key_tuple[0].id
            partial_name = partials_dict[key_tuple]['name']

            if output == self.output_name_id_real:
                partials_block.write(f'{partial_name} = temp_r')
            elif output == self.output_name_id_imag:
                partials_block.write(f'{partial_name} = temp_i')


    def determine_sparse(self):
        return self.determine_sparse_default_elementwise(self.input_size)
