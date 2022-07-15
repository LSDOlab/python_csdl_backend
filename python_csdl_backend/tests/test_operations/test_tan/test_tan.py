from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class TanSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        x = self.create_input('x', val=0.5*np.ones(scalability_param))

        y = csdl.tan(x)

        self.register_output('y', y)


def test_tan():
    nn = 1
    in_val = 0.5*np.ones(nn)
    val = np.tan(in_val)
    vals_dict = {'y': val}

    total = np.diag(1 / ((np.cos(in_val)**2).flatten()))
    totals_dict = {('y', 'x'): total}

    run_test(
        TanSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_tan',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_tan_large():
    nn = (10,10)
    in_val = 0.5*np.ones(nn)
    val = np.tan(in_val)
    vals_dict = {'y': val}

    total = np.diag(1 / ((np.cos(in_val)**2).flatten()))
    totals_dict = {('y', 'x'): total}

    run_test(
        TanSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_tan_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
