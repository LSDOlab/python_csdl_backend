from python_csdl_backend.core.simulator import Simulator
import csdl
import pytest
import numpy as np
import copy


class SampleModel(csdl.Model):

    def define(self):
        x = self.create_input('x')
        y = self.declare_variable('y')
        self.register_output('f', x+1+y)
        self.register_output('z', x-y)

        self.add_objective('f')
        self.add_design_variable('x')
        self.add_constraint('z')

        x = self.create_input('x2', val = np.arange(24).reshape(2,3,4))


def test_simulator_set1():
    m = SampleModel()
    sim = Simulator(m)
    sim['x'] = 1.0
    sim['x2'] = 1.0
    sim['x2'] = np.ones((2,3,4))
    sim['x2'] = np.ones((3,2,4))
    sim['x2'] = np.ones(1)
