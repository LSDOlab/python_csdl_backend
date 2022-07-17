from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_list, get_scalars_list
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.sparse_utils import sparse_matrix

import scipy.sparse as sp
import numpy as np


def get_indexed_passthrough_lite(op):
    return IndexedPassthroughLite


class IndexedPassthroughLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'indexed_passthrough'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.out_name = get_only(self.nx_outputs_dict)
        self.out_var = operation.outs[0]

        # print(self.out_var, self.out_name, self.name)
        self.out_shape = self.out_var.shape
        self.indices = self.out_var._tgt_indices
        self.vals = self.out_var._tgt_vals
        self.out_val = self.out_var.val
        # print()
        # print(self.out_name)
        # print(name)å
        # print(self.indices)

    def get_evaluation(self, eval_block, vars):

        self.out_name_temp = self.out_name+'_'+'_temp'
        # vars[self.out_name_temp] = self.out_val.reshape((self.out_shape))
        vars[self.out_name_temp] = self.out_val
        for in_name_lang, (shape, tgt_indices) in self.indices.items():

            # print('tgt_indice', tgt_indices)
            in_name = self.get_input_id(in_name_lang)
            i = np.unravel_index(tgt_indices, self.out_shape)
            vars[f'i_{in_name}_{self.name}'] = np.unravel_index(tgt_indices, self.out_shape)
            eval_block.write(f'{self.out_name_temp}[i_{in_name}_{self.name}] = {in_name}.flatten()')
            eval_block.write(f'{self.out_name} = {self.out_name_temp}.copy()')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):

        for key_tuple in partials_dict:
            input_id = key_tuple[1].id
            input = self.get_lang_input(input_id)
            output_id = key_tuple[0].id
            partial_name = partials_dict[key_tuple]['name']

            shape, tgt_indices = self.indices[input]
            data = np.ones(len(tgt_indices))
            rows = tgt_indices
            cols = np.arange(len(tgt_indices))

            size = np.prod(shape)
            sizeout = np.prod(self.out_shape)

            if is_sparse_jac:
                vars[partial_name] = sp.csc_matrix((data, (rows, cols)), shape=(sizeout, size))
            else:
                vars[partial_name] = sp.csc_matrix((data, (rows, cols)), shape=(sizeout, size)).toarray()

    def determine_sparse(self):
        # return True
        out_size = np.prod(self.out_shape)
        if out_size < 50:
            return False

        number_of_indices = 0
        in_size = 0
        for in_name_lang, (shape, tgt_indices) in self.indices.items():
            number_of_indices += len(tgt_indices)
            in_size += np.prod(shape)

        sparsity = number_of_indices/(out_size*in_size)
        if sparsity < 0.67:
            # print('index passthrough sparsity:', sparsity)
            return True
        else:
            return False
