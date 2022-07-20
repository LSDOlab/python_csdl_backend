from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class SumSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        w = self.create_input('w', val=3*np.ones(scalability_param))
        x = self.create_input('x', val=2*np.ones(scalability_param))

        y = csdl.sum(w, x)

        self.register_output('y', y)


def test_sum():
    nn = 1
    vals_dict = {'y': 5*np.ones(nn)}
    totals_dict = {
        ('y', 'x'): np.eye(nn),
        ('y', 'w'): np.eye(nn),
    }

    run_test(
        SumSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_sum',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_sum_large():
    nn = (10, 10)
    vals_dict = {'y': 5*np.ones(nn)}
    totals_dict = {
        ('y', 'x'): np.eye(100),
        ('y', 'w'): np.eye(100),
    }

    run_test(
        SumSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_sum_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
