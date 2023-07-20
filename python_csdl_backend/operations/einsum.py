from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_unique_list, get_scalars_list
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.sparse_utils import sparse_matrix
from typing import List, Union, Tuple
from copy import deepcopy

import scipy.sparse as sp
import numpy as np


def get_einsum_lite(op):
    return EinSumLite


class EinSumLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'einsum'
        name = f'{name} {op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.in_names = [var.name for var in operation.dependencies]
        self.in_shapes = [var.shape for var in operation.dependencies]

        # for shape in self.in_shapes:
        #     print(shape)
        # exit()
        self.out_name = operation.outs[0].name
        self.operation_ss = operation.literals['subscripts']
        out_shape = operation.outs[0].shape
        # self.in_vals = [var.val for var in operation.dependencies]

        # Find unused characters in operation
        check_string = 'abcdefghijklmnopqrstuvwxyz'
        self.unused_chars = ''
        for char in check_string:
            if not (char in self.operation_ss):
                self.unused_chars += char

        # Translate the operation string into a list
        self.operation_aslist = []

        # Representation of each tensor in the operation string
        tensor_rep = ''
        for char in self.operation_ss:
            if char.isalpha():
                tensor_rep += char
            elif (char == ',' or char == '-'):
                self.operation_aslist.append(tensor_rep)
                tensor_rep = ''

        # When output is a scalar
        if self.operation_ss[-1] == '>':
            self.operation_aslist.append(tensor_rep)

        # When output is a tensor
        else:
            self.operation_aslist.append(tensor_rep)

        # When output shape is not provided
        if out_shape == None:
            self.out_shape = compute_einsum_shape(
                self.operation_aslist,
                self.in_shapes,
            )
        else:
            self.out_shape = out_shape

        # if self.out_shape == (1, ):
        #     self.add_output(out_name)
        # else:
        #     self.add_output(out_name, shape=self.out_shape)

        completed_in_names = []
        operation_aslist = self.operation_aslist

        self.I = []
        # print(self.op_summary_block.to_string())
        for in_name_index, in_name in enumerate(zip(self.in_names)):
            if in_name in completed_in_names:
                continue
            else:
                completed_in_names.append(in_name)
            # self.add_input(in_name, shape=in_shapes[in_name_index], val=in_val)
            # self.declare_partials(out_name, in_name)

            shape = self.in_shapes[in_name_index]
            size = np.prod(shape)
            rank = len(shape)
            flat_indices = np.arange(size)
            ind = np.unravel_index(flat_indices, shape)
            # self.list_of_tuple_of_indices_of_input_tensors.append(ind)

            # Generate I efficiently for each in_name

            I_shape = 2 * list(shape)
            I_shape = tuple(I_shape)
            I_ind = 2 * list(ind)
            I_ind = tuple(I_ind)

            I = np.zeros(I_shape)
            I[I_ind] += 1
            # print(f'{I.size:,}', f'\t{I.nbytes:,}')

            self.I.append(I)

    def get_evaluation(self, eval_block, vars):

        in_argument = ''
        for in_name in self.in_names:

            in_id = self.get_input_id(in_name)
            in_argument = in_argument + ', '+in_id

        out_id = self.get_output_id(self.out_name)
        eval_block.write(f'{out_id} = np.einsum(\'{self.operation_ss}\' {in_argument})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac, lazy):

        def compute_partials(*input_vals):

            inputs = {}
            out_name = self.out_name
            for i, in_name in enumerate(self.in_names):
                inputs[in_name] = input_vals[i]

            partials = {}

            unused_chars = self.unused_chars
            operation_aslist = self.operation_aslist

            completed_in_names = []

            for in_name_index, in_name in enumerate(self.in_names):
                '''Checking if we are at a repeated input whose derivative was computed at its first occurence in the in_names. If true, we will skip the current iteration of in_name'''
                if in_name in completed_in_names:
                    continue
                else:
                    completed_in_names.append(in_name)

                shape = self.in_shapes[in_name_index]
                size = np.prod(shape)
                rank = len(shape)

                # Compute the locations where the same input is used
                locations = []
                for idx, same_name in enumerate(self.in_names):
                    if same_name == in_name:
                        locations.append(idx)

                new_in_name_tensor_rep = operation_aslist[in_name_index]
                new_in_name_tensor_rep += unused_chars[:rank]
                new_output_tensor_rep = operation_aslist[-1]
                new_output_tensor_rep += unused_chars[:rank]

                new_operation_aslist = deepcopy(operation_aslist)
                new_operation_aslist[in_name_index] = new_in_name_tensor_rep
                new_operation_aslist[-1] = new_output_tensor_rep

                # Compute new_operation by replacing each tensor_rep for in_name in first location by I's tensor_rep
                new_operation = ''
                for string_rep in new_operation_aslist[:-1]:
                    new_operation += string_rep
                    new_operation += ','
                new_operation = new_operation[:-1] + '->'
                new_operation += new_operation_aslist[-1]

                partials[out_name, in_name] = np.einsum(
                    new_operation,
                    *(inputs[in_name] for in_name in self.in_names[:in_name_index]),
                    self.I[in_name_index],
                    *(inputs[in_name] for in_name in self.in_names[in_name_index + 1:]))

                for i in locations[1:]:
                    new_operation_aslist = deepcopy(operation_aslist)
                    new_operation_aslist[
                        i] = operation_aslist[i] + unused_chars[:rank]
                    new_operation_aslist[-1] = new_output_tensor_rep

                    new_operation = ''
                    for string_rep in new_operation_aslist[:-1]:
                        new_operation += string_rep
                        new_operation += ','
                    new_operation = new_operation[:-1] + '->'
                    new_operation += new_operation_aslist[-1]

                    partials[out_name, in_name] += np.einsum(
                        new_operation,
                        *(inputs[in_name] for in_name in self.in_names[:i]),
                        self.I[len(completed_in_names) - 1],
                        *(inputs[in_name]
                          for in_name in self.in_names[i + 1:])).reshape(
                        partials[out_name, in_name].shape)

            partial_val_tuple = []

            for key_tuple in partials_dict:
                input_id = key_tuple[1].id
                input = self.get_lang_input(input_id)

                out_size = np.prod(self.out_shape)
                in_size = np.prod(self.in_shapes[self.in_names.index(input)])
                partial_val = partials[out_name, input].reshape((out_size, in_size))
                partial_val_tuple.append(partial_val)

            return tuple(partial_val_tuple)

        function_name = f'{self.operation.name}_partial_func'
        vars[function_name] = compute_partials

        for i, input_name in enumerate(self.in_names):

            input_id = self.get_input_id(input_name)
            if i == 0:
                argument_in = input_id
            else:
                argument_in = argument_in + ', ' + input_id

        partials_block.write(f'{self.operation.name}_temp_einsum = {function_name}({argument_in})')

        for i, key_tuple in enumerate(partials_dict):
            input = key_tuple[1].id
            output = key_tuple[0].id
            partial_name = partials_dict[key_tuple]['name']
            partials_block.write(f'{partial_name} = {self.operation.name}_temp_einsum[{i}]')



def compute_einsum_shape(
    operation_aslist: List[str],
    in_shapes: Union[Tuple[int], List[Tuple[int]]],
):
    out_shape = []

    if operation_aslist[-1] == '':
        out_shape = (1, )

    else:
        for char in operation_aslist[-1]:
            for i, tensor_rep in enumerate(operation_aslist[:-1]):
                if (char in tensor_rep):
                    shape_ind = tensor_rep.index(char)
                    out_shape.append(in_shapes[i][shape_ind])
                    break

        out_shape = tuple(out_shape)

    return out_shape


def einsum_subscripts_tolist(subscripts: str):
    operation_aslist = []
    tensor_rep = ''
    for char in subscripts:
        if char.isalpha():
            tensor_rep += char
        elif (char == ',' or char == '-'):
            operation_aslist.append(tensor_rep)
            tensor_rep = ''

    # When output is a scalar
    if subscripts[-1] == '>':
        operation_aslist.append(tensor_rep)

    # When output is a tensor
    else:
        operation_aslist.append(tensor_rep)

    return operation_aslist


def new_einsum_subscripts_to_string_and_list(subscripts: List[Tuple], scalar_output=False):
    # Assign characters to each axis_name in the tuples
    unused_chars = 'abcdefghijklmnopqrstuvwxyz'
    axis_map = {}
    operation_as_string = ''
    operation_aslist = []

    if not(scalar_output):
        num_inputs = len(subscripts) - 1
    else:
        num_inputs = len(subscripts)

    for axis_names in subscripts[:num_inputs]:
        tensor_rep = ''

        # Mapping an alphabet for each axis in the tuple
        for axis in axis_names:
            if not (axis in axis_map):
                axis_map[axis] = unused_chars[0]
                unused_chars = unused_chars[1:]
            tensor_rep += axis_map[axis]

        operation_as_string += tensor_rep
        operation_as_string += ','
        operation_aslist.append(tensor_rep)

    tensor_rep = ''

    # When output is a tensor
    if len(subscripts) == (num_inputs + 1):
        for axis in subscripts[-1]:
            tensor_rep += axis_map[axis]

    operation_as_string = operation_as_string[:-1] + '->'
    operation_as_string += tensor_rep
    operation_aslist.append(tensor_rep)

    return operation_aslist, operation_as_string
