import csdl
import numpy as np
import pytest


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


def test_implicit_not_enough_info():
    import python_csdl_backend
    model = Implicit(nlsolver='newton')
    with pytest.raises(NotImplementedError):
        sim = python_csdl_backend.Simulator(model)
