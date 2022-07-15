from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class Log10Sample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        x = self.create_input('x', val=0.5*np.ones(scalability_param))

        y = csdl.log10(x)

        self.register_output('y', y)


def test_log10():
    nn = 1
    vals_dict = {'y': np.log10(0.5*np.ones(nn))}
    totals_dict = {('y', 'x'): np.eye(nn)/(np.log(10)*0.5*np.eye(nn))}

    run_test(
        Log10Sample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_log10',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_log10_large():
    nn = (10, 10)
    vals_dict = {'y': np.log10(0.5*np.ones(nn))}

    total = np.diag(2*np.ones(100)/np.log(10))
    totals_dict = {('y', 'x'): total}

    run_test(
        Log10Sample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_log10_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
