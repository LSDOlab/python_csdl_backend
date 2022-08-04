import csdl
from python_csdl_backend import Simulator
import pytest
import numpy as np


class SampleModel(csdl.Model):
    def define(self):
        x = self.declare_variable('x')
        y = self.declare_variable('y')

        f = csdl.custom(x, y, op=Custom())

        self.register_output('f', f)

        z = self.create_input('z')
        self.add_design_variable('z')  # TEMPORARY


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

        if inputs['x'] < 0:
            raise ValueError('x must be positive')

        derivatives['f', 'x'] = np.ones((1, 1))
        derivatives['f', 'y'] = np.ones((1, 1))


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
    failure_flag = sim.run(check_failure=True)
    assert failure_flag == True


def test_completed_failure_flag():
    sim = Simulator(SampleModel())
    sim.run()
    sim['x'] = 2
    failure_flag = sim.run(check_failure=True)
    assert failure_flag == False


def test_normal_failure_flag():
    sim = Simulator(SampleModel())
    sim.run()
    sim['x'] = 2
    sim.run(check_failure=True)
    np.testing.assert_almost_equal(sim['f'], 3)

    checks = sim.check_partials()
    sim.assert_check_partials(checks)


def test_check_no_failure_flag_deriv():
    m = csdl.Model()
    m.create_input('x')
    m.add(SampleModel())
    m.add_design_variable('x')
    m.add_objective('f')
    sim = Simulator(m)

    sim.run()
    sim['x'] = -1
    sim.run(check_failure=True)
    with pytest.raises(ValueError):
        sim.compute_total_derivatives()


def test_check_failure_flag_deriv():
    m = csdl.Model()
    m.create_input('x')
    m.add(SampleModel())
    m.add_design_variable('x')
    m.add_objective('f')
    sim = Simulator(m)

    sim.run()
    sim['x'] = -1
    sim.run(check_failure=True)
    failure_flag = sim.compute_total_derivatives(check_failure=True)
    assert failure_flag == True


def test_completed_failure_flag_deriv():
    m = csdl.Model()
    m.create_input('x')
    m.add(SampleModel())
    m.add_design_variable('x')
    m.add_objective('f')
    sim = Simulator(m)

    sim.run()
    sim.compute_total_derivatives()
    sim['x'] = 2
    failure_flag = sim.compute_total_derivatives(check_failure=True)
    assert failure_flag == False


def test_normal_failure_flag_deriv():
    m = csdl.Model()
    m.create_input('x')
    m.add(SampleModel())
    m.add_design_variable('x')
    m.add_objective('f')
    sim = Simulator(m)

    sim.run()
    sim.compute_total_derivatives()
    sim['x'] = 2
    sim.run()
    failure_flag = sim.compute_total_derivatives(check_failure=True)

    checks = sim.check_partials()
    sim.assert_check_partials(checks)


if __name__ == '__main__':

    model = SampleModel()
    reo = csdl.GraphRepresentation(model)

    model2 = SampleModel()
    reo2 = csdl.GraphRepresentation(model2)


    # failure_flag = False
    # sim.compute_total_derivatives(check = failure_flag)

    # if failure_flag:
    #     print(failed)