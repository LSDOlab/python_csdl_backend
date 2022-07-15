from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class SinhSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        x = self.create_input('x', val=0.5*np.ones(scalability_param))

        y = csdl.sinh(x)

        self.register_output('y', y)


def test_sinh():
    nn = 1
    in_val = 0.5*np.ones(nn)
    val = np.sinh(in_val)
    vals_dict = {'y': val}

    total = np.diag(np.cosh(in_val).flatten())
    totals_dict = {('y', 'x'): total}

    run_test(
        SinhSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_sinh',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_sinh_large():
    nn = (10, 10)
    in_val = 0.5*np.ones(nn)
    val = np.sinh(in_val)
    vals_dict = {'y': val}

    total = np.diag(np.cosh(in_val).flatten())
    totals_dict = {('y', 'x'): total}

    run_test(
        SinhSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_sinh_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
