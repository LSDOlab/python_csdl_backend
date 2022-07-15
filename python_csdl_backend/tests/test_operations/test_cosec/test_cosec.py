from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class CosecSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        x = self.create_input('x', val=0.5*np.ones(scalability_param))

        y = csdl.cosec(x)

        self.register_output('y', y)


def test_cosec():
    nn = 1
    in_val = 0.5*np.ones(nn)
    val = 1.0/np.sin(in_val)
    vals_dict = {'y': val}

    total = np.diag(-1.0 * (np.cos(in_val) / (np.sin(in_val) * np.sin(in_val))).flatten())

    totals_dict = {('y', 'x'): total}

    run_test(
        CosecSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_cosec',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_cosec_large():
    nn = (10, 10)
    in_val = 0.5*np.ones(nn)
    val = 1.0/np.sin(in_val)
    vals_dict = {'y': val}

    total = np.diag(-1.0 * (np.cos(in_val) / (np.sin(in_val) * np.sin(in_val))).flatten())
    totals_dict = {('y', 'x'):total}

    run_test(
        CosecSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_cosec_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
