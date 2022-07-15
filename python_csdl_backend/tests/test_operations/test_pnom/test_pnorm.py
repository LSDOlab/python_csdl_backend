from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class PNormSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']
        
        x = self.create_input('x', val=2*np.ones(scalability_param))

        y = csdl.pnorm(x, pnorm_type=2)

        self.register_output('y', y)


def test_pnorm():
    nn = 1
    vals_dict = {'y': np.linalg.norm(2*np.ones(nn), ord=2)}
    totals_dict = {}

    run_test(
        PNormSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_pnorm',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_pnorm_large():
    nn = (10, 10)
    vals_dict = {'y': np.linalg.norm(2*np.ones(nn), ord=2)}
    totals_dict = {}

    run_test(
        PNormSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_pnorm_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
