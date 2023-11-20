from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_unique_list, get_scalars_list
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
from csdl.lang.variable import Variable

import numpy as np


def get_exp_a_lite(op):
    if isinstance(op.literals['a'], Variable):
        return ExpAVarLite
    else:
        return ExpALite


class ExpALite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'exp_a'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        input_name_id = get_only(self.nx_inputs_dict)
        output_name_id = get_only(self.nx_outputs_dict)
        self.input_size = np.prod(self.nx_inputs_dict[input_name_id].var.shape)

        self.a = operation.literals['a']

        self.input_name = input_name_id
        self.output_name = output_name_id

        self.a_name = self.name + '_a'

    def get_evaluation(self, eval_block, vars):

        vars[self.a_name] = self.a
        eval_block.write(f'{self.output_name} = {self.a_name}**{self.input_name}')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac, lazy):

        key_tuple = get_only(partials_dict)
        input = key_tuple[1].id
        output = key_tuple[0].id
        partial_name = partials_dict[key_tuple]['name']

        vars[self.a_name] = self.a
        partials_block.write(f'{partial_name} = ({self.a_name}**{input} * np.log({self.a_name})).flatten()')

    def determine_sparse(self):
        return self.determine_sparse_default_elementwise(self.input_size)


class ExpAVarLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'exp_a_var'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        in_names = [var.name for var in operation.dependencies]
        self.in_name_a = in_names[1]
        self.in_name_var = in_names[0]

        self.in_name_a_id = self.input_name_to_unique[self.in_name_a]
        self.in_name_var_id = self.input_name_to_unique[self.in_name_var]

        output_name_id = get_only(self.nx_outputs_dict)
        self.input_size = np.prod(self.nx_inputs_dict[self.in_name_var_id].var.shape)

        self.output_name = output_name_id

    def get_evaluation(self, eval_block, vars):

        eval_block.write(f'{self.output_name} = {self.in_name_a_id}**{self.in_name_var_id}')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac, lazy):

        for key_tuple in partials_dict:
            input = key_tuple[1].id
            output = key_tuple[0].id
            partial_name = partials_dict[key_tuple]['name']

            if input == self.in_name_var_id:
                partials_block.write(f'{partial_name} = ({self.in_name_a_id}**{self.in_name_var_id} * np.log({self.in_name_a_id})).flatten()')
            elif input == self.in_name_a_id:
                partials_block.write(f'{partial_name} = ({self.in_name_var_id}*{self.in_name_a_id}**({self.in_name_var_id}-1.0)).flatten()')

    def determine_sparse(self):
        return self.determine_sparse_default_elementwise(self.input_size)
