from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class CosSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        x = self.create_input('x', val=2*np.ones(scalability_param))

        y = csdl.cos(x)

        self.register_output('y', y)


def test_cos():
    nn = 1
    vals_dict = {'y': np.cos(2*np.ones(nn))}
    totals_dict = {('y', 'x'): np.diag(-np.sin(2*np.ones(nn)))}

    run_test(
        CosSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_cos',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_cos_large():
    nn = (10, 10)
    vals_dict = {'y': np.cos(2*np.ones(nn))}
    totals_dict = {('y', 'x'): np.diag(-np.sin(2*np.ones(100)))}

    run_test(
        CosSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_cos_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


if __name__ == '__main__':
    import python_csdl_backend

    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    sim = python_csdl_backend.Simulator(CosSample(scalability_param=2))
    sim.run()
    sim.check_partials()
    profiler.disable()
    profiler.dump_stats('output')