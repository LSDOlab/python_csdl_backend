from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class SampleSameMult(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        x = self.create_input('x', val=0.5*np.ones(scalability_param))
        self.register_output('y', x*x)


def test_same_mult():
    nn = 1
    vals_dict = {'y': (0.25*np.ones(nn))}
    totals_dict = {('y', 'x'): np.eye(nn)}

    run_test(
        SampleSameMult(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_arccos',
        vals_dict=vals_dict,
        totals_dict=totals_dict)

def test_same_mult_large():
    nn = 10
    vals_dict = {'y': (0.25*np.ones(nn))}
    totals_dict = {('y', 'x'): np.eye(nn)}

    run_test(
        SampleSameMult(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_arccos',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
