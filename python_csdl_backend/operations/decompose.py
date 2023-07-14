from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_unique_list, get_scalars_list
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.sparse_utils import sparse_matrix

import scipy.sparse as sp
import numpy as np


def get_decompose_lite(op):
    return DecomposeLite


class DecomposeLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'decompose'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.in_name = get_only(self.nx_inputs_dict)
        self.invar = self.nx_inputs_dict[self.in_name].var
        self.src_indices = self.operation.src_indices
        self.shape = self.invar.shape
        # self.val = self.invar.val
        self.linear = True

    def get_evaluation(self, eval_block, vars):

        for out_expr, src_indices in self.src_indices.items():
            name = out_expr.name
            shape = out_expr.shape

            src_indices_name = f'src_indices_{name}_{self.operation.name}'
            vars[src_indices_name] = src_indices

            name_id = self.get_output_id(name)
            # eval_block.comment(f'in shape: {self.shape}, in name:{self.in_name}, out shape: {shape}')
            # eval_block.write(f'print(type({self.in_name}))')
            # eval_block.write(f'print({self.in_name}.flatten().shape)')
            eval_block.write(f'{name_id} = (({self.in_name}.flatten())[{src_indices_name}]).reshape({shape})')
            # eval_block.write(f'{name_id} = {self.in_name}.flatten()[{src_indices_name}]')
            # eval_block.write(f'{name_id} = {name_id}.reshape({shape})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac, lazy):

        for key_tuple in partials_dict:
            input = key_tuple[1].id
            output_id = key_tuple[0].id
            partial_name = partials_dict[key_tuple]['name']

            size = np.prod(self.invar.shape)
            sizeout = np.prod(self.nx_outputs_dict[output_id].var.shape)

            for key in self.src_indices:

                output = self.get_lang_output(output_id)
                if output == key.name:
                    temp = self.src_indices[key]

            if lazy:
                cols_name = '_cols'+partial_name
                vars[cols_name] = temp
                
                partials_block.write(f'rows = np.arange({len(temp)})')
                partials_block.write(f'data = np.ones({sizeout})')
                
                if is_sparse_jac:
                    partials_block.write(f'{partial_name} = sp.csc_matrix((data, (rows, {cols_name})), shape=({sizeout}, {size}))')
                else:
                    partials_block.write(f'{partial_name} = sp.csc_matrix((data, (rows, {cols_name})), shape=({sizeout}, {size})).toarray()')
                partials_block.write(f'del rows')
                partials_block.write(f'del data')
            
            else:
                rows = np.arange(len(temp))
                cols = temp
                data = np.ones(sizeout)

                if is_sparse_jac:
                    vars[partial_name] = sp.csc_matrix((data, (rows, cols)), shape=(sizeout, size))
                else:
                    vars[partial_name] = sp.csc_matrix((data, (rows, cols)), shape=(sizeout, size)).toarray()

    def determine_sparse(self):
        # return True
        in_size = np.prod(self.shape)
        if in_size < 50:
            return False

        number_of_indices = 0
        out_size = 0
        for out_expr, src_indices in self.src_indices.items():
            number_of_indices += len(src_indices)
            output_id = self.get_output_id(out_expr.name)
            out_size += np.prod(self.nx_outputs_dict[output_id].var.shape)

        sparsity = number_of_indices/(in_size*out_size)
        if sparsity < 0.67:
            return True
        else:
            return False
