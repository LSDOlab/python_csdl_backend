from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_unique_list, get_scalars_list
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.sparse_utils import sparse_matrix
from python_csdl_backend.utils.custom_utils import (
    check_not_implemented_args,
    process_custom_derivatives_metadata,
    prepare_compute_derivatives,
    postprocess_compute_derivatives,
    is_empty_function,
)

from csdl import CustomExplicitOperation
import scipy.sparse as sp
import numpy as np


def get_custom_explicit_lite(op):
    return CustomExplicitLite


class CustomExplicitLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'custom_explicit'
        name = f'{name}_{op_name}'
        operation.properties = {}
        operation.properties['elementwise'] = False
        # operation.properties = {}
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.ordered_in_names = operation.input_meta
        self.ordered_out_names = operation.output_meta
        self.operation_name = operation.name

        # Object we are passing into the eval and reverse script
        self.explicit_wrapper = CustomExplicitWrapper(operation, self.ordered_in_names, self.ordered_out_names)
        self.wrapper_func_name = f'{name}_func_{list(self.ordered_out_names.keys())[0]}'

        # True if compute_jacvec instead of compute_derivatives
        self.jac_is_function = self.explicit_wrapper.jac_is_function

    def get_evaluation(self, eval_block, vars):

        # For the component evaluation, pass in the wrapped object
        vars[self.wrapper_func_name] = self.explicit_wrapper

        for i, input_name_lang in enumerate(self.ordered_in_names):

            input_name = self.get_input_id(input_name_lang)
            if i == 0:
                arg_in = input_name
            else:
                arg_in = arg_in + ', ' + input_name

        eval_block.write(f'temp = {self.wrapper_func_name}.solve({arg_in})')

        for i, output in enumerate(self.ordered_out_names):

            out_name = self.get_output_id(output)
            eval_block.write(f'{out_name} = temp[{i}].copy()')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac, lazy):

        for key_tuple in partials_dict:
            input_id = key_tuple[1].id
            output_id = key_tuple[0].id
            partial_name = partials_dict[key_tuple]['name']

            # For the component evaluation, If compute_derivatives is given, the jacobian is precompted and pulled.
            output = self.get_lang_output(output_id)
            input = self.get_lang_input(input_id)
            vars[self.wrapper_func_name] = self.explicit_wrapper
            partials_block.write(f'{partial_name} = {self.wrapper_func_name}.get_custom_explicit_partials(\'{output}\', \'{input}\')')

    def get_accumulation_function(self, input_paths, path_output, partials_block, vars):

        # For the component evaluation, If compute_jac_vec is given, compute accumulation as a function of
        # Ouput paths must be in correct order...
        in_argument = ''
        for inv in path_output:
            in_argument += inv+', '
        in_argument = in_argument.rstrip(in_argument[-1])
        in_argument = in_argument.rstrip(in_argument[-1])

        vars[self.wrapper_func_name] = self.explicit_wrapper

        # give paths of outputs to inputs
        partials_block.write(f'{self.operation_name}_path_in = {self.wrapper_func_name}.accumulate_rev({in_argument})')

        # Input paths must be in correct order...
        for i, path in enumerate(input_paths):
            partials_block.write(f'{path} = {self.operation_name}_path_in[{i}].copy()')


class CustomExplicitWrapper():

    def __init__(self, op, ordered_in_names, ordered_out_names):

        self.op = op
        self.derivatives = op.derivatives_meta.copy()
        self.ordered_in_names = ordered_in_names
        self.ordered_out_names = ordered_out_names
        self.needs_partials = True  # is set to true when the last method called is 'compute'

        # Raise error for derivative/input/output parameters not implemented for this backend
        check_not_implemented_args(op, ordered_in_names, 'input')
        check_not_implemented_args(op, ordered_out_names, 'output')
        check_not_implemented_args(op, self.derivatives, 'derivative')

        # process derivative types
        process_custom_derivatives_metadata(
            self.derivatives,
            ordered_out_names,
            ordered_in_names)

        # check if user implemented compute_jacvec or compute_partials
        # Not sure the best way to do this.
        # test_instance = CustomExplicitOperation()
        # if test_instance.compute_derivatives.__func__ is op.compute_derivatives.__func__:
        if is_empty_function(op.compute_derivatives.__func__):

            # if test_instance.compute_jacvec_product.__func__ is not op.compute_jacvec_product.__func__:
            if not is_empty_function(op.compute_jacvec_product.__func__):
                self.use_compute_jacvec = True
            else:
                all_deriv_vals_given = True
                for derivative_tuple in self.derivatives:
                    if self.derivatives[derivative_tuple]['backend_type'] != 'row_col_val_given':
                        all_deriv_vals_given = False

                if all_deriv_vals_given:
                    self.use_compute_jacvec = False
                else:
                    raise ValueError(f'either compute_jacvec_product or compute_derivatives must be defined in {op}')
        else:
            # if test_instance.compute_jacvec_product.__func__ is not op.compute_jacvec_product.__func__:
            if not is_empty_function(op.compute_jacvec_product.__func__):
                # If both compute_jacvec and compute_derivatives are overwritten, use copmute_derivatives for now
                self.use_compute_jacvec = False
            else:
                self.use_compute_jacvec = False

        # print(self.use_compute_jacvec, self.op.name)

        self.jac_is_function = self.use_compute_jacvec

    def solve(self, *input_vals):

        # set inputs and outputs. TODO: preprocess as much as we can
        self.inputs = {}
        for i, input_name in enumerate(self.ordered_in_names):
            self.inputs[input_name] = (input_vals[i].copy()).reshape(self.ordered_in_names[input_name]['shape'])
        outputs = {}
        for i, output_name in enumerate(self.ordered_out_names):
            outputs[output_name] = np.zeros(self.ordered_out_names[output_name]['shape'])

        # run compute method
        self.op.compute(self.inputs.copy(), outputs)

        # compute derivatives beforehand
        if not self.jac_is_function:
            self.needs_partials = True

        # process outputs
        output_tuple = []
        for i, output_name in enumerate(self.ordered_out_names):

            if isinstance(outputs[output_name], np.ndarray):
                output_tuple.append(outputs[output_name].copy())
            else:
                output_tuple.append(np.array(outputs[output_name]))
        output_tuple = tuple(output_tuple)

        # return tuple of numpy arrays
        return output_tuple

    def get_custom_explicit_partials(self, output_name, input_name):

        if self.needs_partials:
            self.totals = self.compute_derivatives(self.inputs)
            self.needs_partials = False

        # Gets called if compute_derivatives is given instead of compute_JVP.
        # Derivatives are already computed in evaluation so no need to compute it.
        return self.totals[output_name, input_name]

    def compute_derivatives(self, inputs):

        derivatives = {}

        # Set derivatives
        derivatives = prepare_compute_derivatives(self.derivatives)

        # run user-defined method
        self.op.compute_derivatives(inputs, derivatives)

        # Post-process user given derivatives
        postprocess_compute_derivatives(derivatives, self.derivatives)
        return derivatives

    def accumulate_rev(self, *output_paths):

        # This method is called only for compute JVP
        d_outs = {}
        for i, out in enumerate(self.ordered_out_names):
            d_outs[out] = output_paths[i]

            # num_rows should be equal for each out in d_outs
            num_rows = d_outs[out].shape[0]

        # initialize d_inputs as zeros
        # We will slowly build this row by row, called compute_JVP for each one.
        accumulated_paths_dict = {}
        for input_name in self.ordered_in_names:
            in_size = np.prod(self.ordered_in_names[input_name]['shape'])
            accumulated_paths_dict[input_name] = np.zeros((num_rows, in_size))

        # build each row
        for row_index in range(num_rows):
            # initial d_outputs: users ARE NOT writing to d_ouputs
            d_outputs = {}
            for output_name in d_outs:
                out_shape = self.ordered_out_names[output_name]['shape']

                if isinstance(d_outs[output_name], np.ndarray):
                    d_outputs[output_name] = d_outs[output_name][row_index, :].reshape(out_shape)
                else:
                    d_outputs[output_name] = (d_outs[output_name][row_index, :].toarray()).reshape(out_shape)

            # initial d_inputs: users ARE writing to d_inputs
            d_inputs = {}
            for input_name in self.ordered_in_names:
                in_shape = self.ordered_in_names[input_name]['shape']
                d_inputs[input_name] = np.zeros(in_shape)

            # self.inputs is computed from initial model evaluation.
            # users ARE NOT writing to self.inputs
            self.op.compute_jacvec_product(self.inputs, d_inputs, d_outputs, 'rev')

            # Replace each row of initialized path accumulation
            for input_name in self.ordered_in_names:
                accumulated_paths_dict[input_name][row_index, :] = d_inputs[input_name].flatten()

        # Return in correct order.
        accumulated_paths = []
        for input in self.ordered_in_names:
            accumulated_paths.append(accumulated_paths_dict[input])

        # paths are now solved
        accumulated_paths = tuple(accumulated_paths)
        return accumulated_paths
