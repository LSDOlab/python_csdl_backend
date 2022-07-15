from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class LinearCombinationSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        w = self.create_input('w', val=3*np.ones(scalability_param))
        x = self.create_input('x', val=2*np.ones(scalability_param))

        y = 2*w+4*x

        self.register_output('y', y)


def test_linear_combination():
    nn = 1
    vals_dict = {'y': 6*np.ones(nn) + 8*np.ones(nn)}
    totals_dict = {
        ('y', 'x'): 4*np.eye(nn),
        ('y', 'w'): 2*np.eye(nn),
    }

    run_test(
        LinearCombinationSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_linear_combination',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_linear_combination_large():
    nn = (10, 10)
    vals_dict = {'y': 6*np.ones(nn) + 8*np.ones(nn)}
    totals_dict = {
        ('y', 'x'): 4*np.eye(100),
        ('y', 'w'): 2*np.eye(100),
    }

    run_test(
        LinearCombinationSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_linear_combination_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
