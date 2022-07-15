from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class PNormAxisSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        x = self.create_input('x', val=2*np.ones(scalability_param))

        y = csdl.pnorm(x, pnorm_type=2, axis=1)

        self.register_output('y', y)


def test_pnorm_axis_large():
    nn = (10, 5)
    vals_dict = {'y': np.linalg.norm(2*np.ones(nn), axis=1, ord=2)}
    totals_dict = {}

    run_test(
        PNormAxisSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_pnorm_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


class PNormAxisSample2(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        x = self.create_input('x', val=2*np.ones(scalability_param))

        y = csdl.pnorm(x, pnorm_type=2, axis=0)

        self.register_output('y', y)


def test_pnorm_axis_large2():
    nn = (10, 5)
    vals_dict = {'y': np.linalg.norm(2*np.ones(nn), axis=0, ord=2)}
    totals_dict = {}

    run_test(
        PNormAxisSample2(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_pnorm_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
