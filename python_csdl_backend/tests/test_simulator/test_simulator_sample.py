from python_csdl_backend.core.simulator import Simulator
import csdl
import pytest
import numpy as np


class SampleModel(csdl.Model):
    def define(self):
        x = self.create_input('x')
        y = self.declare_variable('y')
        self.register_output('f', x+1+y)


class SampleModel2(csdl.Model):
    def define(self):
        x = self.create_input('x')

        m = csdl.Model()
        y = m.declare_variable('x')
        z = m.register_output('y', y*3+y)
        self.add(m, name = 'm')

        z1 = self.declare_variable('y')

        self.register_output('f', x+1+z1)

# TODO: add test to raise error if simulator mode argument is wrong

# TODO: add test to raise error if simulator model argument is wrong

# TODO: add a bunch of tests to make sure model.eval_graph is directed, acyclical does not contain subgraphs, etc..


def test_check_partials():
    sim = Simulator(SampleModel())

    sim.run()
    sim.check_partials()


def test_check_totals():
    sim = Simulator(SampleModel())

    sim.run()
    t = sim.check_totals()
    np.testing.assert_almost_equal(len(t), 2)


def test_check_totals_of_wrt():
    sim = Simulator(SampleModel())
    sim.run()
    t = sim.check_totals(of='f', wrt='x')
    np.testing.assert_almost_equal(len(t), 1)


def test_check_totals_no_wrt():
    sim = Simulator(SampleModel())
    sim.run()

    with pytest.raises(KeyError):
        sim.check_totals(of='f', wrt='z')


def test_check_totals_no_of():
    sim = Simulator(SampleModel())
    sim.run()

    t = sim.check_totals(of='y', wrt='x')
    np.testing.assert_almost_equal(len(t), 1)

def test_check_totals_no_of():
    sim = Simulator(SampleModel2())
    sim.run()

    t = sim.check_totals(of='y', wrt='x')
    np.testing.assert_almost_equal(len(t), 1)