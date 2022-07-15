from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class ExpandSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        x = self.create_input('x', val=2*np.ones(scalability_param))

        y = csdl.expand(x, (20, 6))

        self.register_output('y', y)


def test_expand():
    nn = 1
    vals_dict = {'y': 2*np.ones((20, 6))}
    totals_dict = {('y', 'x'): np.ones((120, 1))}

    run_test(
        ExpandSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_expand',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


class ExpandSampleTensor(csdl.Model):


    def define(self):

        val = np.array([
            [1., 2., 3.],
            [4., 5., 6.],
        ])

        x = self.create_input('x', val=val)

        y = csdl.expand(x, (2, 4, 3, 1), indices='ij->iajb')

        self.register_output('y', y)


def test_expand_large():
    array = np.array([
        [1., 2., 3.],
        [4., 5., 6.],
    ])
    expanded_array = np.empty((2, 4, 3, 1))
    for i in range(4):
        for j in range(1):
            expanded_array[:, i, :, j] = array

    vals_dict = {'y': expanded_array}
    totals_dict = {}

    run_test(
        ExpandSampleTensor(),
        outs=['y'],
        ins=['x'],
        name='test_expand_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
