from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class ArcSinSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        x = self.create_input('x', val=0.5*np.ones(scalability_param))

        y = csdl.arcsin(x)

        self.register_output('y', y)


def test_arcsin():
    nn = 1
    vals_dict = {'y': np.arcsin(0.5*np.ones(nn))}
    totals_dict = {('y', 'x'): np.eye(nn)/np.sqrt(np.eye(nn)-(0.5*np.eye(nn))**2)}

    run_test(
        ArcSinSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_arcsin',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_arcsin_large():
    nn = (10, 10)
    vals_dict = {'y': np.arcsin(0.5*np.ones(nn))}
    totals_dict = {('y', 'x'): np.diag(np.ones(100)/np.sqrt(np.ones(100)-(0.5*np.ones(100))**2))}

    run_test(
        ArcSinSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_arcsin_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
