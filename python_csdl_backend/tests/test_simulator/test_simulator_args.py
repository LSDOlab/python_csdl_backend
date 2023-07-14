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

def test_simulator_args():
    m = SampleModel()
    sim = Simulator(m)

def test_opt_totals():
    m = SampleModel()
    sim = Simulator(m)

    sim.run()
    sim.compute_total_derivatives()

    np.testing.assert_almost_equal(sim.objective_gradient(), 1.0)
    np.testing.assert_almost_equal(sim.constraint_jacobian(), 1.0)


# def test_check_totals():
#     m = SampleModel()
#     sim = Simulator(m)

#     sim.run()
#     t = sim.check_totals()
#     np.testing.assert_almost_equal(len(t), 2)


# def test_check_totals_of_wrt():
#     sim = Simulator(SampleModel())
#     sim.run()
#     t = sim.check_totals(of='f', wrt='x')
#     np.testing.assert_almost_equal(len(t), 1)


# def test_check_totals_no_wrt():
#     sim = Simulator(SampleModel())
#     sim.run()

#     with pytest.raises(KeyError):
#         sim.check_totals(of='f', wrt='z')


# def test_check_totals_no_of():
#     sim = Simulator(SampleModel())
#     sim.run()

#     with pytest.raises(KeyError):
#         sim.check_totals(of='y', wrt='x')
