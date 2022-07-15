from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class CotanhSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        x = self.create_input('x', val=0.5*np.ones(scalability_param))

        y = csdl.cotanh(x) + 1

        self.register_output('y', y)


def test_cotanh():
    nn = 1
    in_val = 0.5*np.ones(nn)
    val = 1/np.tanh(in_val) + 1
    vals_dict = {'y': val}

    total = np.diag(1.0 - ((1.0 / np.tanh(in_val))**2).flatten())
    totals_dict = {('y', 'x'): total}

    run_test(
        CotanhSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_cotanh',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_cotanh_large():
    nn = (10, 10)
    in_val = 0.5*np.ones(nn)
    val = 1/np.tanh(in_val) + 1
    vals_dict = {'y': val}

    total = np.diag(1.0 - ((1.0 / np.tanh(in_val))**2).flatten())
    totals_dict = {('y', 'x'): total}

    run_test(
        CotanhSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_cotanh_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
