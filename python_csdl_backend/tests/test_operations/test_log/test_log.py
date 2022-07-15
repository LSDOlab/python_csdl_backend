from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class LogSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        x = self.create_input('x', val=0.5*np.ones(scalability_param))

        y = csdl.log(x)

        self.register_output('y', y)


def test_log():
    nn = 1
    vals_dict = {'y': np.log(0.5*np.ones(nn))}
    totals_dict = {('y', 'x'): np.eye(nn)/(0.5*np.eye(nn))}

    run_test(
        LogSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_log',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_log_large():
    nn = (10, 10)
    vals_dict = {'y': np.log(0.5*np.ones(nn))}

    total = np.diag(2*np.ones(100))
    totals_dict = {('y', 'x'): total}

    run_test(
        LogSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_log_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
