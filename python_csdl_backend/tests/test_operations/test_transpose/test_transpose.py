from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class TransposeSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        x = self.create_input('x', val=0.5*np.ones(scalability_param))

        y = csdl.transpose(x)

        self.register_output('y', y)


def test_transpose():
    nn = 1
    in_val = 0.5*np.ones(nn)
    val = np.transpose(in_val)
    vals_dict = {'y': val}

    total = np.eye(nn)
    totals_dict = {('y', 'x'): total}

    run_test(
        TransposeSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_transpose',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_transpose_large():
    nn = (10,10)
    in_val = 0.5*np.ones(nn)
    val = np.transpose(in_val)
    vals_dict = {'y': val}

    totals_dict = {}

    run_test(
        TransposeSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_transpose_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
