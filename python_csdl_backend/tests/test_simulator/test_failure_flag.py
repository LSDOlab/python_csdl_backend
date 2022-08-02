import csdl
from python_csdl_backend import Simulator
import pytest
import numpy as np

class SampleModel(csdl.Model):
    def define(self):
        x = self.create_input('x')
        y = self.declare_variable('y')

        f = csdl.custom(x, y, op = Custom())

        self.register_output('f', f)

class Custom(csdl.CustomExplicitOperation):

    def define(self):
        self.add_input('x')
        self.add_input('y')
        self.add_output('f')

        self.declare_derivatives('*', '*')

    def compute(self, inputs, outputs):

        if inputs['x'] < 0:
            raise ValueError('x must be positive')

        outputs['f'] = inputs['x'] + inputs['y']

    def compute_derivatives(self, inputs, derivatives):

        derivatives['f', 'x'] =  np.ones((1,1))
        derivatives['f', 'y'] =  np.ones((1,1))


def test_check_no_failure_flag():
    sim = Simulator(SampleModel())
    sim.run()
    sim['x'] = -1
    with pytest.raises(ValueError):
        sim.run()

def test_check_failure_flag():
    sim = Simulator(SampleModel())
    sim.run()
    sim['x'] = -1
    completed = sim.run(failure_flag=True)
    assert completed == False


def test_completed_failure_flag():
    sim = Simulator(SampleModel())
    sim.run()
    sim['x'] = 2
    completed = sim.run(failure_flag=True)
    assert completed == True

def test_normal_failure_flag():
    sim = Simulator(SampleModel())
    sim.run()
    sim['x'] = 2
    sim.run(failure_flag=True)
    np.testing.assert_almost_equal(sim['f'], 3)

    checks = sim.check_partials()
    sim.assert_check_partials(checks)
