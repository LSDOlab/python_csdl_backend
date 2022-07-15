from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class ConcatenationSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        x = self.create_input('x', val=2*np.ones(scalability_param))

        if isinstance(scalability_param, tuple):
            new_shape = (2*scalability_param[0], 2*scalability_param[1])
        else:
            new_shape = (2*scalability_param, )

        y = self.create_output('y', shape=new_shape)

        if isinstance(scalability_param, tuple):
            y[0, 0] = x[0, 0]
            y[0, scalability_param[1]-1] = x[scalability_param[0]-1, scalability_param[1]-1]
        else:
            y[1] = x[0]


def test_concatenation():
    nn = 1
    val = np.ones(2)
    val[1] = 2.0
    vals_dict = {'y': val}

    deriv = np.zeros((2, 1))
    deriv[1, 0] = 1.0
    totals_dict = {('y', 'x'): deriv}

    run_test(
        ConcatenationSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_concatenation',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_concatenation_large():
    nn = (10, 10)

    val = np.ones((20, 20))
    val[0, 0] = 2.0
    val[0, 9] = 2.0

    vals_dict = {'y': val}

    deriv = np.zeros((400, 100))
    deriv[0, 0] = 1.0
    deriv[9, 99] = 1.0
    totals_dict = {('y', 'x'): deriv}

    run_test(
        ConcatenationSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_concatenation_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


if __name__ == '__main__':

    test_concatenation()
