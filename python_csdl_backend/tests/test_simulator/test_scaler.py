from python_csdl_backend.core.simulator import Simulator
import csdl
import pytest
import numpy as np
import copy


class SampleModel2(csdl.Model):

    def define(self):
        x = self.create_input('x1', val=np.ones((3, 2)))
        y = self.declare_variable('y', val=np.ones((3, 2)))
        self.register_output('f1', csdl.pnorm(x+1+y))
        self.register_output('z1', x-y)

        self.add_objective('f1', scaler=10.0)
        self.add_design_variable('x1', scaler=np.ones((3, 2))*100)
        self.add_constraint('z1', scaler=11.0, lower=-5.0)


def test_scaler():
    sim = Simulator(SampleModel2())

    sim.run()

    np.testing.assert_almost_equal(sim.get_design_variable_metadata()['x1']['scaler'], np.ones((3, 2))*100)
    np.testing.assert_almost_equal(sim.obj['scaler'], np.ones((1,))*10.0)
    np.testing.assert_almost_equal(sim.get_constraints_metadata()['z1']['scaler'], np.ones((3, 2))*(11.0))
    np.testing.assert_almost_equal(sim.get_constraints_metadata()['z1']['lower'], np.ones((3, 2))*(-55.0))
