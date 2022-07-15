from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class PowerCombinationSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        w = self.create_input('w', val=3*np.ones(scalability_param))
        x = self.create_input('x', val=2*np.ones(scalability_param))

        y = w*x**2

        self.register_output('y', y)


def test_power_combination():
    nn = 1
    vals_dict = {'y': 12*np.ones(nn)}
    totals_dict = {
        ('y', 'x'): 12*np.eye(nn),
        ('y', 'w'): 4*np.eye(nn),
    }

    run_test(
        PowerCombinationSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_power_combination',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_power_combination_large():
    nn = (10, 10)
    vals_dict = {'y': 12*np.ones(nn)}
    totals_dict = {
        ('y', 'x'): 12*np.eye(100),
        ('y', 'w'): 4*np.eye(100),
    }

    run_test(
        PowerCombinationSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_power_combination_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
