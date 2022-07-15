from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class RotmatSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        x = self.create_input('x', val=0.5*np.ones(scalability_param))

        y = csdl.rotmat(x, axis='x')

        self.register_output('y', y)


def test_rotmat():
    nn = 1
    in_val = 0.5*np.ones(nn)
    # val = np.rotmat(in_val)
    val = in_val
    vals_dict = {}

    total = np.eye(nn)
    totals_dict = {}

    run_test(
        RotmatSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_rotmat',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_rotmat_large():
    nn = (2, )
    vals_dict = {}

    totals_dict = {}

    run_test(
        RotmatSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_rotmat_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
