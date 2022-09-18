from python_csdl_backend.tests.create_single_test import run_test
from python_csdl_backend.tests.test_operations.test_custom_implicit.custom_implicit_solutions import fwd_assertion, deriv_assertion
import csdl
import numpy as np


class CustomImp(csdl.CustomImplicitOperation):

    def initialize(self):
        self.parameters.declare('nlsolver')

    def define(self):

        self.add_input('a', shape=(2, 2))
        self.add_input('b')
        self.add_input('c')

        self.add_output('x', shape=(2,))
        self.add_output('y')

        self.declare_derivatives('x', 'a')
        self.declare_derivatives('x', 'b')
        self.declare_derivatives('x', 'c')
        self.declare_derivatives('x', 'x')
        self.declare_derivatives('x', 'y')

        self.declare_derivatives('y', 'y', rows=[0], cols=[0], val=[1.0])

        nlsolver = self.parameters['nlsolver']
        if self.parameters['nlsolver'] == 'nlbgs':
            self.nonlinear_solver = csdl.NonlinearBlockGS(maxiter=100)
        elif self.parameters['nlsolver'] == 'newton':
            self.nonlinear_solver = csdl.NewtonSolver(solve_subsystems=True)
        else:
            raise ValueError(f'solver {nlsolver} not found')

        self.linear_solver = csdl.ScipyKrylov()

    def evaluate_residuals(self, inputs, outputs, residuals):

        x = outputs['x'][0]
        a = inputs['a'][1, 1]
        b = inputs['b']
        c = inputs['c']

        residuals['x'][0] = x + a*b*x**2 + c*b
        residuals['x'][1] = outputs['x'][1] - 1.0

        residuals['y'] = outputs['y'] - 2.0

    def compute_derivatives(self, inputs, outputs, derivatives):
        print('CD CALL')

        x = outputs['x'][0]
        a = inputs['a'][1, 1]
        b = inputs['b']
        c = inputs['c']

        dxdx = np.zeros((2, 2))
        dxdx[0, 0] = 1.0 + 2.0*x*a*b
        dxdx[1, 1] = 1.0
        derivatives['x', 'x'] = dxdx

        dxda = np.zeros((2, 4))
        dxda[0, 3] = 1*b*x**2
        derivatives['x', 'a'] = dxda

        dxdc = np.zeros((2, 1))
        dxdc[0, 0] = b
        derivatives['x', 'c'] = dxdc

        dxdb = np.zeros((2, 1))
        dxdb[0, 0] = c+a*x**2
        derivatives['x', 'b'] = dxdb

        self.inv_jac_x = np.linalg.inv(dxdx)
        self.inv_jac_y = np.ones((1,))

    # def apply_inverse_jacobian(self, d_outputs, d_residuals, mode):
    #     print('AIJ CALL: mode', mode)
    #     if mode == 'rev':
    #         d_residuals['x'] = (self.inv_jac_x.T).dot(d_outputs['x'])
    #         d_residuals['y'] = (self.inv_jac_y.T)*d_outputs['y']
    #     if mode == 'fwd':
    #         d_outputs['x'] = self.inv_jac_x.dot(d_residuals['x'])
    #         d_outputs['y'] = self.inv_jac_y*d_residuals['y']

    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        print('JVP CALL: mode', mode, '\td_in:', list(d_inputs.keys()), '\td_out:', list(d_outputs.keys()), '\td_res:', list(d_residuals.keys()))
        if mode == 'fwd':
            x = outputs['x'][0]
            a = inputs['a'][1, 1]
            b = inputs['b']
            c = inputs['c']

            dxdx = np.zeros((2, 2))
            dxdx[0, 0] = 1.0 + 2.0*x*a*b
            dxdx[1, 1] = 1.0
            if 'x' in d_outputs:
                d_residuals['x'] += (dxdx).dot(d_outputs['x'])

            dxda = np.zeros((2, 4))
            dxda[0, 3] = 1*b*x**2
            if 'a' in d_inputs:
                d_residuals['x'] += ((dxda).dot(d_inputs['a'].flatten())).reshape(d_residuals['x'].shape)

            dxdc = np.zeros((2, 1))
            dxdc[0, 0] = b
            if 'c' in d_inputs:
                d_residuals['x'] += ((dxdc).dot(d_inputs['c'].flatten())).reshape(d_residuals['x'].shape)

            dxdb = np.zeros((2, 1))
            dxdb[0, 0] = c+a*x**2
            if 'b' in d_inputs:
                d_residuals['x'] += ((dxdb).dot(d_inputs['b'].flatten())).reshape(d_residuals['x'].shape)

        if mode == 'rev':
            x = outputs['x'][0]
            a = inputs['a'][1, 1]
            b = inputs['b']
            c = inputs['c']

            dxdx = np.zeros((2, 2))
            dxdx[0, 0] = 1.0 + 2.0*x*a*b
            dxdx[1, 1] = 1.0
            if 'x' in d_outputs:
                d_outputs['x'] += (dxdx.T).dot(d_residuals['x'])

            dxda = np.zeros((2, 4))
            dxda[0, 3] = 1*b*x**2
            if 'a' in d_inputs:
                d_inputs['a'] += ((dxda.T).dot(d_residuals['x'])).reshape(d_inputs['a'].shape)

            dxdc = np.zeros((2, 1))
            dxdc[0, 0] = b
            if 'c' in d_inputs:
                d_inputs['c'] += ((dxdc.T).dot(d_residuals['x'])).reshape(d_inputs['c'].shape)

            dxdb = np.zeros((2, 1))
            dxdb[0, 0] = c+a*x**2
            if 'b' in d_inputs:
                d_inputs['b'] += ((dxdb.T).dot(d_residuals['x'])).reshape(d_inputs['b'].shape)


class Implicit(csdl.Model):
    def initialize(self):
        self.parameters.declare('nlsolver')

    def define(self):

        nlsolver = self.parameters['nlsolver']

        a_val = 3*np.ones((2, 2))
        a_val[1, 1] = 1.5
        a = self.create_input('a', val=a_val)
        b = self.create_input('b', val=1/2.0)
        c = self.create_input('c', val=-1.0)
        x, y = csdl.custom(a, b, c, op=CustomImp(nlsolver=nlsolver))

        self.register_output('x', x)
        self.register_output('y', y)
        self.register_output('f', x[0]+y)


def test_implicit_newton_jvp():
    vals_dict = fwd_assertion
    totals_dict = deriv_assertion
    run_test(
        Implicit(nlsolver='newton'),
        outs=['f', 'x', 'y'],
        ins=['a', 'b', 'c'],
        vals_dict=vals_dict,
        totals_dict=totals_dict,
    )


def test_implicit_nlbgs_jvp():
    vals_dict = fwd_assertion
    totals_dict = deriv_assertion
    run_test(
        Implicit(nlsolver='nlbgs'),
        outs=['f', 'x', 'y'],
        ins=['a', 'b', 'c'],
        vals_dict=vals_dict,
        totals_dict=totals_dict,
    )


if __name__ == '__main__':
    from python_csdl_backend import Simulator
    # from csdl_om import Simulator

    sim = Simulator(Implicit(nlsolver='newton'), mode='rev')
    sim.run()
    print(sim['f'])  # [2.38742589]
    print('START')
    # print(sim.executable.compute_totals(of='x', wrt=['a', 'b', 'c']))
    # sim.executable.check_totals(of='f', wrt=['a', 'b', 'c'], compact_print=True)
    # sim.check_totals(of='x', wrt=['a', 'b', 'c'], compact_print=True)
    # sim.check_totals(of='x', wrt=['a', 'b', 'c'], compact_print=True)
    # sim.check_partials()

    tots = sim.compute_totals(of=['f', 'x', 'y'], wrt=['a', 'b', 'c'])
    for key in tots:
        print(key)
        print(tots[key])
        print()

    # sim.check_partials(compact_print=True)
    # 'f'                            wrt 'a'                            | 9.4931e-02 | 4.7476e-02 | 4.7455e-02 | 9.9956e-01
    # 'f'                            wrt 'b'                            | 9.8012e-01 | 4.8994e-01 | 4.9018e-01 | 1.0005e+00
    # 'f'                            wrt 'c'                            | 6.3246e-01 | 3.1625e-01 | 3.1621e-01 | 9.9988e-01
