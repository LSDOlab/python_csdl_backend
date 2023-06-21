from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_list, get_scalars_list
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.sparse_utils import sparse_matrix

import scipy.sparse as sp
import numpy as np


def get_reshape_lite(op):
    return ReshapeLite


class ReshapeLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'reshape'
        name = f'{name} {op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)
        self.elementwise = True
        self.linear = True

        self.in_name = get_only(self.nx_inputs_dict)
        self.invar = self.operation.dependencies[0]

        self.out_name = get_only(self.nx_outputs_dict)
        self.outvar = self.operation.outs[0]

        self.shape = self.invar.shape
        self.outshape = self.outvar.shape
        self.size = np.prod(self.shape)

        # self.val = self.invar.val

    def get_evaluation(self, eval_block, vars):
        eval_block.write(f'{self.out_name} = {self.in_name}.reshape({self.outshape})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):

        key_tuple = get_only(partials_dict)
        partial_name = partials_dict[key_tuple]['name']

        # OLD:
        # if not is_sparse_jac:
        #     vars[partial_name] = np.eye(self.size)
        # else:

        # NEW:
        # if elementwise, pass in 1-d diagonal
        vars[partial_name] = np.ones(self.size)

    def determine_sparse(self):
        return self.determine_sparse_default_elementwise(self.size)
