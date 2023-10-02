from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class SampleSameKwargs(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        x = self.create_input('x', val=0.5*np.ones((scalability_param,scalability_param)))

        self.register_output('y', csdl.sum(x,x,x,x,x, axes = (0,)))


def test_same_kwargs():
    nn = 1
    vals_dict = {'y': (2.5*np.ones(nn))}
    totals_dict = {('y', 'x'): 5*np.eye(nn)}

    run_test(
        SampleSameKwargs(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_arccos',
        vals_dict=vals_dict,
        totals_dict=totals_dict)

def test_same_kwargs_large():
    nn = 3
    vals_dict = {'y': (7.5*np.ones(nn))}
    totals_dict = {('y', 'x'): np.array([[5., 0., 0., 5., 0., 0., 5., 0., 0.], [0., 5., 0., 0., 5., 0., 0., 5., 0.], [0., 0., 5., 0., 0., 5., 0., 0., 5.]])}

    run_test(
        SampleSameKwargs(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_arccos',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
