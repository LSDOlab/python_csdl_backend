import numpy as np
from scipy import linalg
import scipy.sparse as sp


class ImplicitSolverCustomBase():

    def __init__(self, op, ins, outs):

        self.op = op
        self.ordered_inputs = ins
        self.ordered_outs = outs
        self.derivatives = op.derivatives_meta

        self.tol = 1e-10

        self.inputs = {}
        for input_name in self.ordered_inputs:
            self.inputs[input_name] = op.input_meta[input_name]

        self.states = {}
        self.total_state_size = 0
        for state in self.ordered_outs:
            self.states[state] = op.output_meta[state]

            self.states[state]['index_lower'] = self.total_state_size
            self.states[state]['size'] = np.prod(self.states[state]['shape'])
            self.total_state_size += self.states[state]['size']
            self.states[state]['index_upper'] = self.total_state_size

            # Keep state initial guess for implicit operation ????
            self.states[state]['initial_val'] = np.ones(self.states[state]['shape'])

        self.residuals = {}
        for state in self.states:
            self.residuals[state] = {}
            self.residuals[state]['state'] = state
        # Raise error for derivative/input/output parameters not implemented for this backend
        not_implemented_input = {
            'src_indices': None,
            'flat_src_indices': None,
            'units': None,
            'desc': '',
            'tags': None,
            'shape_by_conn': False,
            'copy_shape': None
        }

        not_implemented_output = {
            'res_units': None,
            'lower': None,
            'units': None,
            'desc': '',
            'tags': None,
            'shape_by_conn': False,
            'copy_shape': None,
            'upper': None,
            'ref': 1.0,
            'ref0': 0.0,
            'res_ref': 1.0,
            'distributed': None,
        }

        not_implemented_derivatives = {
            'method': 'exact',
            'step': None,
            'form': None,
            'step_calc': None,
        }

        for input in self.inputs:
            temp = self.inputs[input]

            for key_dont in not_implemented_input:
                if temp[key_dont] != not_implemented_input[key_dont]:
                    raise NotImplementedError(f'\'{key_dont}\' for CustomExplicitOperation has not been implemeted in this backend. input \'{input}\' in {type(op)} cannot be processed.')

        for output in self.states:
            temp = self.states[output]

            for key_dont in not_implemented_output:
                if temp[key_dont] != not_implemented_output[key_dont]:
                    raise NotImplementedError(f'\'{key_dont}\' for CustomExplicitOperation has not been implemeted in this backend. output \'{output}\' in {type(op)} cannot be processed.')

        for derivative in self.derivatives:
            temp = self.derivatives[derivative]

            for key_dont in not_implemented_derivatives:
                if temp[key_dont] != not_implemented_derivatives[key_dont]:
                    raise NotImplementedError(f'\'{key_dont}\' for CustomExplicitOperation has not been implemeted in this backend. derivative \'{derivative}\' in {type(op)} cannot be processed.')

        # process derivative types
        for derivative_tuple in self.derivatives:
            given_rows = self.derivatives[derivative_tuple]['rows']
            given_cols = self.derivatives[derivative_tuple]['cols']
            given_val = self.derivatives[derivative_tuple]['val']

            of_var = derivative_tuple[0]
            wrt_var = derivative_tuple[1]

            size_out = np.prod(self.states[of_var]['shape'])
            if wrt_var in self.inputs:
                size_in = np.prod(self.inputs[wrt_var]['shape'])
            else:
                size_in = np.prod(self.states[wrt_var]['shape'])

            if given_rows is not None and given_cols is not None:
                if given_val is None:
                    self.derivatives[derivative_tuple]['backend_type'] = 'row_col_given'
                elif given_val is not None:
                    self.derivatives[derivative_tuple]['backend_type'] = 'row_col_val_given'
                    self.derivatives[derivative_tuple]['given_val'] = sp.csc_matrix((given_val, (given_rows, given_cols)), shape=(size_out, size_in))
            else:
                self.derivatives[derivative_tuple]['backend_type'] = 'standard'

        for out_str in self.ordered_outs:
            for in_str in self.ordered_inputs:
                derivative_tuple = (out_str, in_str)
                if derivative_tuple not in self.derivatives:

                    print(derivative_tuple)
                    size_out = np.prod(self.states[out_str]['shape'])
                    size_in = np.prod(self.inputs[in_str]['shape'])

                    self.derivatives[derivative_tuple] = {}
                    self.derivatives[derivative_tuple]['backend_type'] = 'row_col_val_given'
                    self.derivatives[derivative_tuple]['given_val'] = sp.csc_matrix((size_out, size_in))
            for out_str2 in self.ordered_outs:
                derivative_tuple = (out_str, out_str2)
                if derivative_tuple not in self.derivatives:

                    print(derivative_tuple)
                    size_out = np.prod(self.states[out_str]['shape'])
                    size_in = np.prod(self.states[out_str2]['shape'])

                    self.derivatives[derivative_tuple] = {}
                    self.derivatives[derivative_tuple]['backend_type'] = 'row_col_val_given'
                    self.derivatives[derivative_tuple]['given_val'] = sp.csc_matrix((size_out, size_in))

        # store vals here
        self.state_vals = {}
        self.res_vals = {}
        self.input_vals = {}

    def solve(self, *inputs):

        # Set inputs to the implicit operation:
        for i, input_name in enumerate(self.ordered_inputs):
            self.input_vals[input_name] = inputs[i]

        # Set initial guess:
        for state_name in self.states:
            self.state_vals[state_name] = self.states[state_name]['initial_val']

        # All the initial values are now set. Solve residuals.
        self._solve_implicit()

        # Set and return solved states and exposed variables
        return_tuple = []
        for output in self.states:
            return_tuple.append(self.state_vals[output])
        return_tuple = tuple(return_tuple)
        return return_tuple

    def prepare_evaluate_residuals(self):

        for input_name in self.input_vals:
            self.input_vals[input_name] = self.input_vals[input_name].reshape(self.inputs[input_name]['shape'])

        for output_name in self.state_vals:
            self.state_vals[output_name] = self.state_vals[output_name].reshape(self.states[output_name]['shape'])

        residuals = {}
        for residuals_name in self.states:
            residuals[residuals_name] = np.zeros(self.states[residuals_name]['shape'])

        return self.input_vals, self.state_vals, residuals

    def prepare_totals(self):

        # Set derivatives
        derivatives = {}
        for derivative_tuple in self.derivatives:

            # If rows and cols are given, give a flat vector with size len(rows) or size len(cols)
            if self.derivatives[derivative_tuple]['backend_type'] == 'row_col_given':
                len_val = len(self.derivatives[derivative_tuple]['rows'])
                derivatives[derivative_tuple] = np.zeros((len_val, ))
            else:

                # Otherwise, give zeros of 2D jac matrix
                size_out = np.prod(self.states[derivative_tuple[0]]['shape'])
                if derivative_tuple[1] in self.ordered_inputs:
                    size_in = np.prod(self.inputs[derivative_tuple[1]]['shape'])
                else:
                    size_in = np.prod(self.states[derivative_tuple[1]]['shape'])

                derivatives[derivative_tuple] = np.zeros((size_out, size_in))

        return derivatives

    def build_jac(self, totals_dict):
        '''
        Build thhe block matrix jacobian of residuals wrt states given as totals_dict
        '''

        # DENSE/SPARSE
        # self.residual_jac = np.zeros((self.total_state_size, self.total_state_size))
        self.residual_jac = sp.csc_matrix((self.total_state_size, self.total_state_size))

        for tuple_key in totals_dict:

            if (tuple_key[0] not in self.residuals) or (tuple_key[1] not in self.states):
                continue

            state_of = self.residuals[tuple_key[0]]['state']
            state_wrt = tuple_key[1]

            il_o = self.states[state_of]['index_lower']
            iu_o = self.states[state_of]['index_upper']
            il_w = self.states[state_wrt]['index_lower']
            iu_w = self.states[state_wrt]['index_upper']

            self.residual_jac[il_o:iu_o, il_w:iu_w] = totals_dict[tuple_key]

    def process_totals(self, derivatives):

        # Post-process user given derivatives
        for derivative_tuple in self.derivatives:

            size_out = np.prod(self.states[derivative_tuple[0]]['shape'])
            if derivative_tuple[1] in self.states:
                size_in = np.prod(self.states[derivative_tuple[1]]['shape'])
            else:
                size_in = np.prod(self.inputs[derivative_tuple[1]]['shape'])

            if self.derivatives[derivative_tuple]['backend_type'] == 'row_col_val_given':
                # If the value is given in define, use that.
                derivatives[derivative_tuple] = self.derivatives[derivative_tuple]['given_val']
            elif self.derivatives[derivative_tuple]['backend_type'] == 'row_col_given':

                # If the rows and cols are given, create sparse matrix of only vals.
                given_rows = self.derivatives[derivative_tuple]['rows']
                given_cols = self.derivatives[derivative_tuple]['cols']
                derivatives[derivative_tuple] = sp.csc_matrix((derivatives[derivative_tuple], (given_rows, given_cols)), shape=(size_out, size_in))
            else:
                # If standard derivative, just use user-given derivatie directly.
                derivatives[derivative_tuple] = derivatives[derivative_tuple].reshape((size_out, size_in))

        return derivatives
