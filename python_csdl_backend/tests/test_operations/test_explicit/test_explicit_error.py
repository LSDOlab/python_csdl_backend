from python_csdl_backend.tests.create_single_test import run_test
import python_csdl_backend
from csdl import CustomExplicitOperation
import csdl
import numpy as np
import pytest


class Example(CustomExplicitOperation):
    def define(self):
        self.add_input('b', src_indices=2)
        self.add_input('a')

        self.add_output('x')
        self.add_output('y')

        self.declare_derivatives('y', 'b')
        self.declare_derivatives('y', 'a')
        self.declare_derivatives('x', 'a')
        self.declare_derivatives('x', 'b')

    def compute(self, inputs, outputs):
        outputs['x'] = inputs['a'] + inputs['b']
        outputs['y'] = inputs['b']*inputs['a']

    def compute_derivatives(self, inputs, derivatives):
        derivatives['y', 'b'] = inputs['a']
        derivatives['y', 'a'] = inputs['b']
        derivatives['x', 'b'] = np.ones((1, 1))
        derivatives['x', 'a'] = np.ones((1, 1))


class ExplicitRun(csdl.Model):

    def define(self):

        a = self.create_input('a', val=2.0)
        b = self.create_input('b', val=3.0)
        x, y = csdl.custom(b, a, op=Example())

        self.register_output('x', x)
        self.register_output('y', y)
        self.register_output('f', x+1)


def test_explicit_error():
    with pytest.raises(NotImplementedError):
        sim = python_csdl_backend.Simulator(ExplicitRun())
