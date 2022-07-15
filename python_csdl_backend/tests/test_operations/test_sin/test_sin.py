from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class SinSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        x = self.create_input('x', val=2*np.ones(scalability_param))

        y = csdl.sin(x)

        self.register_output('y', y)


def test_sin():
    nn = 1
    vals_dict = {'y': np.sin(2*np.ones(nn))}
    totals_dict = {('y', 'x'): np.diag(np.cos(2*np.ones(nn)))}

    run_test(
        SinSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_sin',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_sin_large():
    nn = (10, 10)
    vals_dict = {'y': np.sin(2*np.ones(nn))}
    totals_dict = {('y', 'x'): np.diag(np.cos(2*np.ones(100)))}

    run_test(
        SinSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_sin_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


if __name__ == '__main__':

    test_sin()
