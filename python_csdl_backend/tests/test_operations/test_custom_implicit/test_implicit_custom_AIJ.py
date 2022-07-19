from python_csdl_backend.tests.create_single_test import run_test
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

    def evaluate_residuals(self, inputs, outputs, residuals):

        x = outputs['x'][0]
        a = inputs['a'][1, 1]
        b = inputs['b']
        c = inputs['c']

        residuals['x'][0] = x + a*b*x**2 + c*b
        residuals['x'][1] = outputs['x'][1] - 1.0

        residuals['y'] = outputs['y'] - 2.0

    def compute_derivatives(self, inputs, outputs, derivatives):

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

    def apply_inverse_jacobian(self, d_outputs, d_residuals, mode):
        if mode == 'rev':
            d_residuals['x'] = (self.inv_jac_x.T).dot(d_outputs['x'])
            d_residuals['y'] = (self.inv_jac_y.T)*d_outputs['y']
        if mode == 'fwd':
            d_outputs['x'] = self.inv_jac_x.dot(d_residuals['x'])
            d_outputs['y'] = self.inv_jac_y*d_residuals['y']


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


def test_implicit_newton_aij():
    vals_dict = {}
    totals_dict = {}
    run_test(
        Implicit(nlsolver='newton'),
        outs=['f', 'x', 'y'],
        ins=['a', 'b', 'c'],
        vals_dict=vals_dict,
        totals_dict=totals_dict,
    )
