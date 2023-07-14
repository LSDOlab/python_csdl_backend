

from python_csdl_backend.operations.implicit.wrappers.wrapper_base import ImplicitWrapperBase
from python_csdl_backend.utils.custom_utils import (
    check_not_implemented_args,
    process_custom_derivatives_metadata,
    prepare_compute_derivatives,
    postprocess_compute_derivatives,
    is_empty_function,
)
from csdl import CustomImplicitOperation

import numpy as np


class ImplicitCustomWrapper(ImplicitWrapperBase):

    def __init__(self, op, ins, outs):

        self.op = op
        self.ordered_inputs = ins
        self.ordered_outs = outs
        self.derivatives = op.derivatives_meta.copy()

        self.tol = 1e-10

        self.inputs = {}
        for input_name in self.ordered_inputs:
            self.inputs[input_name] = op.input_meta[input_name]
            self.inputs[input_name]['size'] = np.prod(self.inputs[input_name]['shape'])

        self.states = {}
        self.total_state_size = 0
        for state in self.ordered_outs:
            self.states[state] = op.output_meta[state]
            # print(state, op.output_meta[state])
            # exit()

            self.states[state]['index_lower'] = self.total_state_size
            self.states[state]['size'] = np.prod(self.states[state]['shape'])
            self.total_state_size += self.states[state]['size']
            self.states[state]['index_upper'] = self.total_state_size

            # Keep state initial guess for implicit operation ????
            # self.states[state]['initial_val'] = np.ones(self.states[state]['shape'])
            self.states[state]['initial_val'] = self.states[state]['val'].reshape(self.states[state]['shape'])

        self.residuals = {}
        for state in self.states:
            self.residuals[state] = {}
            self.residuals[state]['state'] = state
            self.states[state]['residual'] = state

        self.exposed = {}

        # Process user declared meta-data
        check_not_implemented_args(op, self.inputs, 'input')
        check_not_implemented_args(op, self.states, 'output')
        check_not_implemented_args(op, self.derivatives, 'derivative')

        # process derivative types
        process_custom_derivatives_metadata(
            self.derivatives,
            self.states,
            {**self.inputs, **self.states})

        # store vals here
        self.state_vals = {}
        self.res_vals = {}
        self.input_vals = {}

        # check which user defined methods have been given
        # test_instance = CustomImplicitOperation()
        self.CD_given = False
        if not is_empty_function(op.compute_derivatives.__func__):
            self.CD_given = True
        self.CJVP_given = False
        if not is_empty_function(op.compute_jacvec_product.__func__):
            self.CJVP_given = True
        self.AIJ_given = False
        if not is_empty_function(op.apply_inverse_jacobian.__func__):
            self.AIJ_given = True

        # determine which combination of user-defined derivative methods we use:
        # -- compute_derivatives (CD)
        # -- apply_inverse_jacobians (AIJ)
        # -- compute_jacvec_product (CJVP)
        # in order of priority/efficiency:
        # 1) if CJVP and AIJ    -->     pure matrix-free method
        # 2.a) if CJVP and CD   -->     manual residual/state jac inverse/solve
        # 2.b) if AIJ and CD    -->     manual residual/input jac matmat
        # 3.a) if CD only       -->     pure matrix method
        # 3.b) if CJVP only     -->     manual residual/state jac inverse/solve but pure matrix-free for residual/input jac
        # 5) if none of ^       -->     not enough information to solve.
        # 3a more efficient than 3b?

        if self.CJVP_given and self.AIJ_given:
            self.res_inverse_type = 'AIJ'
            self.res_input_type = 'CJVP'
        elif self.CJVP_given and self.CD_given:
            self.res_inverse_type = 'CD'
            self.res_input_type = 'CJVP'
        elif self.AIJ_given and self.CD_given:
            self.res_inverse_type = 'AIJ'
            self.res_input_type = 'CD'
        elif self.CD_given:
            self.res_inverse_type = 'CD'
            self.res_input_type = 'CD'
        elif self.CJVP_given:
            self.res_inverse_type = 'CJVP'
            self.res_input_type = 'CJVP'
        else:
            raise NotImplementedError(f'Not enough methods given to compute derivatives of custom implicit operation {op}')

        # raise not implemented error
        if self.res_inverse_type == 'CJVP':
            raise NotImplementedError(f'jacvec product for residual state jac not yet implemented.')

    def prepare_evaluate_residuals(self):

        for input_name in self.input_vals:
            self.input_vals[input_name] = self.input_vals[input_name].reshape(self.inputs[input_name]['shape'])

        for output_name in self.state_vals:
            # print(output_name, self.state_vals[output_name])
            self.state_vals[output_name] = self.state_vals[output_name].reshape(self.states[output_name]['shape'])

        residuals = {}
        for residuals_name in self.states:
            residuals[residuals_name] = np.zeros(self.states[residuals_name]['shape'])

        return self.input_vals, self.state_vals, residuals

    def run(self):
        # compute residual
        self.input_vals, self.state_vals, self.res_vals = self.prepare_evaluate_residuals()
        self.op.evaluate_residuals(
            self.input_vals,
            self.state_vals,
            self.res_vals,
        )

    def solve_residuals(self):

        for input_name in self.input_vals:
            self.input_vals[input_name] = self.input_vals[input_name].reshape(self.inputs[input_name]['shape'])

        for output_name in self.state_vals:
            self.state_vals[output_name] = np.zeros((self.states[output_name]['shape']))

        self.op.solve_residual_equations(self.input_vals, self.state_vals)

    def set_input(self, input_name, val):
        self.input_vals[input_name] = val

    def get_input_size(self, input_name):
        return np.prod(self.input_vals[input_name].shape)

    def set_state(self, state_name, val):
        self.state_vals[state_name] = val

    def get_residual(self, res_name):
        return self.res_vals[res_name]

    def get_state(self, state_name):
        return self.state_vals[state_name]

    def compute_totals(self):

        totals = prepare_compute_derivatives(self.derivatives)
        self.op.compute_derivatives(
            self.input_vals,
            self.state_vals,
            totals,
        )
        postprocess_compute_derivatives(totals, self.derivatives)

        return totals

    def apply_inverse_jac(self, d_residuals):

        # precompute d_residuals
        applied_inverse_jac = {}
        for res_name in d_residuals:
            shape = self.states[res_name]['shape']
            applied_inverse_jac[res_name] = np.zeros(shape)

        # compute the inverse jacobian
        self.op.apply_inverse_jacobian(
            applied_inverse_jac,
            d_residuals,
            'fwd',
        )
        return applied_inverse_jac

    def apply_inverse_jacT(self, d_outputs):

        # precompute d_residuals
        applied_inverse_jacT = {}
        for res_name in d_outputs:
            shape = self.states[res_name]['shape']
            applied_inverse_jacT[res_name] = np.zeros(shape)

        # compute the inverse jacobian
        self.op.apply_inverse_jacobian(
            d_outputs,
            applied_inverse_jacT,
            'rev',
        )
        return applied_inverse_jacT

    def compute_rev_jvp(self, d_r, d_in, d_o):
        inputs = self.input_vals.copy()
        outputs = self.state_vals.copy()
        self.op.compute_jacvec_product(inputs, outputs, d_in, d_o, d_r, 'rev')
        return d_in, d_o
