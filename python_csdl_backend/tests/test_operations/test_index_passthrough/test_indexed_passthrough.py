from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class IndexedPassthroughSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        x = self.create_input('x', val=2*np.ones(scalability_param))
        x1 = csdl.reshape(x, new_shape=scalability_param + (1, ))
        x2 = 3.0*x1

        y = self.create_output('y', shape=scalability_param + (3, ))
        y[:, :, 0] = x1
        y[:, :, 1] = x2


def test_indexed_passthrough():
    nn = (1, 1)
    val = np.zeros((1, 1, 3))
    val[:, :, 0] = 2
    val[:, :, 1] = 6
    val[:, :, 2] = 1

    vals_dict = {'y': val}
    totals_dict = {('y', 'x'): np.array([[1.], [3.], [0]])}

    run_test(
        IndexedPassthroughSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_indexed_passthrough',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_indexed_passthrough_large():
    nn = (5, 8)
    val = np.zeros((5, 8, 3))
    val[:, :, 0] = 2
    val[:, :, 1] = 6
    val[:, :, 2] = 1

    vals_dict = {'y': val}
    totals_dict = {}

    run_test(
        IndexedPassthroughSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_sin_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
