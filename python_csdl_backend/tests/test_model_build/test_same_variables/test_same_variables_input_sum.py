from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class SampleSameSum(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        x = self.create_input('x', val=0.5*np.ones(scalability_param))

        self.register_output('y', csdl.sum(x,x,x,x,x))


def test_same_sum():
    nn = 1
    vals_dict = {'y': (2.5*np.ones(nn))}
    totals_dict = {('y', 'x'): 5*np.eye(nn)}

    run_test(
        SampleSameSum(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_arccos',
        vals_dict=vals_dict,
        totals_dict=totals_dict)

def test_same_sum_large():
    nn = 10
    vals_dict = {'y': (2.5*np.ones(nn))}
    totals_dict = {('y', 'x'): 5*np.eye(nn)}

    run_test(
        SampleSameSum(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_arccos',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
