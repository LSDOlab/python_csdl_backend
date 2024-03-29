from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class ArctanSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        x = self.create_input('x', val=2*np.ones(scalability_param))

        y = csdl.arctan(x)

        self.register_output('y', y)


def test_arctan():
    nn = 1
    vals_dict = {'y': np.arctan(2*np.ones(nn))}
    totals_dict = {('y', 'x'): np.eye(nn)/(np.eye(nn)+(2*np.eye(nn))**2)}

    run_test(
        ArctanSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_arctan',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_arctan_large():
    nn = (10, 10)
    vals_dict = {'y': np.arctan(2*np.ones(nn))}
    totals_dict = {('y', 'x'): np.diag(np.ones(100)/(np.ones(100)+(2*np.ones(100))**2))}

    run_test(
        ArctanSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_arctan_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
