from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class PrintVarSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        x = self.create_input('x', val=0.5*np.ones(scalability_param))

        self.print_var(x)
        y = csdl.tanh(x) + 1

        self.register_output('y', y)


def test_print_var():
    nn = 1
    in_val = 0.5*np.ones(nn)
    val = np.tanh(in_val) + 1
    vals_dict = {'y': val}

    total = np.diag((1.0 - np.tanh(in_val)**2).flatten())
    totals_dict = {('y', 'x'): total}

    run_test(
        PrintVarSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_print_var',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_print_var_large():
    nn = (2, 2)
    in_val = 0.5*np.ones(nn)
    val = np.tanh(in_val) +1 
    vals_dict = {'y': val}

    total = np.diag((1.0 - np.tanh(in_val)**2).flatten())
    totals_dict = {('y', 'x'): total}

    run_test(
        PrintVarSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_print_var_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
