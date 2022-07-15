from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_list, get_scalars_list
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
import numpy as np
import scipy.sparse as sp


def get_power_combination_lite(op):
    return PowerCombinationLite


class PowerCombinationLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'power_combination'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.in_names = [d.name for d in self.operation.dependencies]
        self.out_name = operation.outs[0].name
        self.out_shape = operation.outs[0].shape
        self.coeff = self.operation.literals['coeff']
        self.coeff_name = operation.name+'_coeff'
        self.powers = get_scalars_list(self.operation.literals['powers'], self.in_names)

        self.coeff_val = self.coeff

        self.input_size = np.prod(self.operation.dependencies[0].shape)

    def get_evaluation(self, eval_block, vars):

        vars[self.coeff_name] = self.coeff_val
        output_name = self.get_output_id(self.out_name)

        i = 0
        # for in_name_lang, power in zip(self.in_names, self.powers):
        #     in_name = self.get_input_id(in_name_lang)
        #     eval_block.write(f'print(\'{in_name}\',{in_name}.shape)')

        for in_name_lang, power in zip(self.in_names, self.powers):

            in_name = self.get_input_id(in_name_lang)

            if i == 0:
                eval_block.write(f'{output_name} = ({in_name}**{power})')
            else:
                eval_block.write(f'*({in_name}**{power})', linebreak=False)

            i += 1
        # eval_block.write(f'print({self.coeff_name})')
        # eval_block.write(f'print(\'{output_name}\',{output_name}.shape)')
        eval_block.write(f'{output_name} = ({output_name}*{self.coeff_name}).reshape({self.out_shape})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac):

        for key_tuple in partials_dict:
            input = key_tuple[1].id
            output = key_tuple[0].id
            partial_name = partials_dict[key_tuple]['name']

            input_lang = self.get_lang_input(input)
            power = self.powers[self.in_names.index(input_lang)]
            shape = self.out_shape
            size = np.prod(shape)
            coeff_name = self.coeff_name+'_temp'
            vars[coeff_name] = self.coeff_val*np.ones(shape)

            partial_ind_name = partial_name+'_inds'
            vars[partial_name+'_inds'] = tuple(np.arange(size))
            if is_sparse_jac:
                pass
            else:
                vars[partial_name] = np.eye(size)

            partials_block.write(f'temp_power = {coeff_name}')
            for in_name, power in zip(self.in_names, self.powers):
                a = 1.
                b = power
                if in_name == input_lang:
                    a = power
                    b = power - 1.

                if b == 0.0:
                    partials_block.write(f'*{a}', linebreak=False)
                else:
                    if (a == 1.) and (b == 1.):
                        partials_block.write(f'*({self.get_input_id(in_name)})', linebreak=False)
                    elif a == 1.:
                        partials_block.write(f'*({self.get_input_id(in_name)}**{b})', linebreak=False)
                    elif b == 1.:
                        partials_block.write(f'*{a}*({self.get_input_id(in_name)})', linebreak=False)
                    else:
                        partials_block.write(f'*{a}*({self.get_input_id(in_name)}**{b})', linebreak=False)

            if is_sparse_jac:
                partials_block.write(f'{partial_name} = sp.csc_matrix((temp_power.flatten(), ({partial_ind_name},{partial_ind_name})))')
            else:
                partials_block.write(f'{partial_name}[{partial_name}_inds,{partial_name}_inds]  = temp_power.flatten()')

    def determine_sparse(self):
        return self.determine_sparse_default_elementwise(self.input_size)
