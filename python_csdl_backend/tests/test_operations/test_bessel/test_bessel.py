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
        x = self.create_input('x', val=scalability_param)
        y = csdl.bessel(x, kind=kind, order = order)
        self.register_output('y', y)

def test_bessel_1_1():
    nn = (10,10)
    val = np.arange(100).reshape(nn)+1
    vals_dict = {'y': jv(1,val)}
    totals_dict = {('y', 'x'): np.diag(jvp(1,val).flatten())}

    run_test(
        BesselSample(
            scalability_param=val,
            kind = 1,
            order = 1,),
        outs=['y'],
        ins=['x'],
        name='test_bessel',
        vals_dict=vals_dict,
        totals_dict=totals_dict)

def test_bessel_2_2():
    nn = (10,10)
    val = np.arange(100).reshape(nn)+1
    vals_dict = {'y': yv(2,val)}
    totals_dict = {('y', 'x'): np.diag(yvp(2,val).flatten())}

    run_test(
        BesselSample(
            scalability_param=val,
            kind = 2,
            order = 2,),
        outs=['y'],
        ins=['x'],
        name='test_bessel',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_bessel_2_2_orders():
    nn = (10,10)
    val = np.arange(100).reshape(nn)+1
    orders = np.arange(100).reshape(nn)+5
    vals_dict = {'y': yv(orders,val)}
    totals_dict = {('y', 'x'): np.diag(yvp(orders,val).flatten())}


    run_test(
        BesselSample(
            scalability_param=val,
            kind = 2,
            order = orders,),
        outs=['y'],
        ins=['x'],
        name='test_bessel',
        vals_dict=vals_dict,
        totals_dict=totals_dict)

def test_bessel_error():
    import pytest
    nn = (10,10)
    val = np.arange(100).reshape(nn)+1

    import csdl
    with pytest.raises(Exception):
        gr = csdl.GraphRepresentation(BesselSample(
                scalability_param=val,
                kind = 1.5,
                order = 1.5,))


def test_bessel_order_shape_error():
    import pytest
    val = np.arange(100)
    orders = np.arange(99)

    with pytest.raises(Exception):
        gr = csdl.GraphRepresentation(BesselSample(
                scalability_param=val,
                kind = 1,
                order = orders,))

if __name__ == '__main__':
    test_bessel_2_2_orders()