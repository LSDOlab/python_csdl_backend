from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_unique_list, get_scalars_list
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
import numpy as np


def get_hankel_lite(op):
    return HankelLite


class HankelLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'hankel'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        input_name_id = get_only(self.nx_inputs_dict)
        output_name_id = get_only(self.nx_outputs_dict)
        self.input_size = np.prod(self.nx_inputs_dict[input_name_id].var.shape)

        self.input_name = input_name_id
        self.output_name = output_name_id
        self.kind = operation.literals['kind']
        self.order = operation.literals['order']
        self.func_name = self.name
        self.func_name_partials = self.name+"_partial_func"

    def get_evaluation(self, eval_block, vars):

        if self.kind == 1:
            from scipy.special import hankel1
            vars[self.func_name] = hankel1
        elif self.kind == 2:
            from scipy.special import hankel2
            vars[self.func_name] = hankel2
        # print (self.order)
        # print (self.kind)
        # print (vars[self.func_name])
        eval_block.write(f'{self.output_name}={self.func_name}({self.order},{self.input_name})')
        # eval_block.write(f'{self.output_name} = np.sin({self.input_name})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac, lazy):
        
        key_tuple = get_only(partials_dict)
        input = key_tuple[1].id
        output = key_tuple[0].id
        partial_name = partials_dict[key_tuple]['name']

        # OLD FULL JACOBIAN
        # if is_sparse_jac:
        #     partials_block.write(f'{partial_name} = sp.diags(np.cos({input}).flatten(), format = \'csc\')')
        # else:
        #     partials_block.write(f'{partial_name} = np.diag(np.cos({input}).flatten())')

        # NEW: 
        # only return diag values for elementwise
        # Also sparsity doesn't matter
        if self.kind == 1:
            from scipy.special import h1vp
            vars[self.func_name_partials] = h1vp
        elif self.kind == 2:
            from scipy.special import h2vp
            vars[self.func_name_partials] = h2vp
        partials_block.write(f'{partial_name} = {self.func_name_partials}({self.order},{self.input_name}).flatten()')

    def determine_sparse(self):
        return self.determine_sparse_default_elementwise(self.input_size)