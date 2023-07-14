from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_unique_list, get_scalars_list
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
import numpy as np
import scipy.sparse as sp


def get_sparsematmat_lite(op):
    return SparsematmatLite


class SparsematmatLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'sparsematmat'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        shape = operation.dependencies[0].shape
        in_name = operation.dependencies[0].name
        out_name = operation.outs[0].name
        self.sparse_mat_eval = operation.literals['sparse_mat']
        self.in_name = self.get_input_id(in_name)
        self.out_name = self.get_output_id(out_name)
        self.linear = True

        # val = operation.dependencies[0].val
        self.sparse_mat = operation.literals['sparse_mat']

        self.num_sparse_rows = self.sparse_mat.shape[0]
        self.num_sparse_cols = self.sparse_mat.shape[1]

        output_shape = self.num_sparse_rows, shape[1]

        num_inputs = np.prod(shape)
        num_outputs = np.prod(output_shape)

        # A_data = self.sparse_mat.data
        A_data = self.sparse_mat[self.sparse_mat.nonzero()]
        A_rows, A_cols = self.sparse_mat.nonzero()

        row_indices = np.arange(num_outputs).reshape(output_shape)
        col_indices = np.arange(num_inputs).reshape(shape)

        vals = np.outer(A_data, np.ones(shape[1]))
        rows = row_indices[A_rows]
        cols = col_indices[A_cols]
        print(vals.flatten(), vals.flatten().shape)
        print(rows.flatten(), rows.flatten().shape)
        print(cols.flatten(), cols.flatten().shape)
        print(self.full_namespace)

        self.sparse_mat_deriv = sp.csc_matrix((vals.flatten(), (rows.flatten(), cols.flatten())), shape=(num_outputs, num_inputs))

    def get_evaluation(self, eval_block, vars):

        sparse_mat_name = self.name+'_mat'
        vars[sparse_mat_name] = self.sparse_mat_eval
        str = f'{self.out_name} = {sparse_mat_name}@{self.in_name}'
        eval_block.write(str)

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac, lazy):

        key_tuple = get_only(partials_dict)
        input = key_tuple[1].id
        output = key_tuple[0].id
        partial_name = partials_dict[key_tuple]['name']

        if is_sparse_jac:
            vars[partial_name] = self.sparse_mat_deriv
        else:
            vars[partial_name] = self.sparse_mat_deriv.toarray()

    def determine_sparse(self):
        return True
