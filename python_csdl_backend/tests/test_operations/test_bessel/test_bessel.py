from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np

from scipy.special import jv, jvp, yv, yvp

class BesselSample(csdl.Model):
    def initialize(self):
        self.parameters.declare('scalability_param')
        self.parameters.declare('kind')
        self.parameters.declare('order')

    def define(self):
        scalability_param = self.parameters['scalability_param']
        kind = self.parameters['kind']
        order = self.parameters['order']

        # orde
        x = self.create_input('x', val=2*np.ones(scalability_param))
        y = csdl.bessel(x, kind=kind, order = order)
        self.register_output('y', y)

def test_bessel_1_1():
    nn = (10,10)
    vals_dict = {'y': jv(1,2*np.ones(nn))}
    totals_dict = {('y', 'x'): np.diag(jvp(1,2*np.ones(nn)).flatten())}

    run_test(
        BesselSample(
            scalability_param=nn,
            kind = 1,
            order = 1,),
        outs=['y'],
        ins=['x'],
        name='test_bessel',
        vals_dict=vals_dict,
        totals_dict=totals_dict)

def test_bessel_2_2():
    nn = (10,10)
    vals_dict = {'y': yv(2,2*np.ones(nn))}
    totals_dict = {('y', 'x'): np.diag(yvp(2,2*np.ones(nn)).flatten())}

    run_test(
        BesselSample(
            scalability_param=nn,
            kind = 2,
            order = 2,),
        outs=['y'],
        ins=['x'],
        name='test_bessel',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
    

def test_bessel_error():
    import pytest
    nn = (10,10)

    import csdl
    with pytest.raises(Exception):
        gr = csdl.GraphRepresentation(BesselSample(
                scalability_param=nn,
                kind = 1.5,
                order = 1.5,))