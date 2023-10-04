from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class DecomposeSample1(csdl.Model):

    def initialize(self):
        self.parameters.declare('val')

    def define(self):
        val = self.parameters['val']
        # x = self.create_input('x', val=val)
        # self.register_output('y', x[2:4,0:3,0])

        shape1 = (3,)
        x = self.create_input('x', val=2*np.ones(shape1))
        x1 = csdl.reshape(x, new_shape=shape1 + (1, ))
        x2 = 3.0*x1

        y = self.create_output('y', shape=shape1 + (3, ))
        y[:, 0] = x1
        # y[:, 1] = x2

def test_decompose_sample_1():
    x = np.arange(120).reshape(5,4,6)

    vals_dict = {'y': x}
    totals_dict = {('y', 'x'): np.eye(120)}

    run_test(
        DecomposeSample1(val=x),
        outs=['y'],
        ins=['x'],
        name='test_indexed_passthrough',
        vals_dict=vals_dict,
        totals_dict=totals_dict)

if __name__ == '__main__':
    import python_csdl_backend
    import csdl
    x = np.arange(120).reshape(5,4,6)

    vals_dict = {'y': x}
    totals_dict = {('y', 'x'): np.eye(120)}
    r = csdl.GraphRepresentation(DecomposeSample1(val=x))