from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class CotanSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        x = self.create_input('x', val=0.5*np.ones(scalability_param))

        y = csdl.cotan(x)

        self.register_output('y', y)


def test_cotan():
    nn = 1
    in_val = 0.5*np.ones(nn)
    val = 1.0/np.tan(in_val)
    vals_dict = {'y': val}

    total = np.diag(-1.0 / (np.sin(in_val)**2).flatten())

    totals_dict = {('y', 'x'): total}

    run_test(
        CotanSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_cotan',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_cotan_large():
    nn = (10, 10)
    in_val = 0.5*np.ones(nn)
    val = 1.0/np.tan(in_val)
    vals_dict = {'y': val}

    total = np.diag(-1.0 / (np.sin(in_val)**2).flatten())

    totals_dict = {('y', 'x'): total}

    run_test(
        CotanSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_cotan_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
