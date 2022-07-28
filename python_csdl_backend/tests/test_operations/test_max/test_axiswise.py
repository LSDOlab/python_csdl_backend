from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class AxismaxSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        m = 2
        n = 3
        o = 4
        p = 5
        q = 6

        # Shape of a tensor
        tensor_shape = (m, n, o, p, q)

        num_of_elements = np.prod(tensor_shape)
        # Creating the values of the tensor
        val = np.arange(num_of_elements).reshape(tensor_shape)

        # Declaring the tensor as an input
        ten = self.declare_variable('x', val=val)

        # Computing the axiswise maximum on the tensor
        axis = 1
        ma = self.register_output('y', csdl.max(ten, axis=axis))


def test_axismax():
    m = 2
    n = 3
    o = 4
    p = 5
    q = 6

    tensor_shape = (m, n, o, p, q)
    num_of_elements = np.prod(tensor_shape)
    tensor = np.arange(num_of_elements).reshape(tensor_shape)

    val = np.amax(tensor, axis=1)
    vals_dict = {'y': val}

    # total = np.eye(nn)
    totals_dict = {}

    run_test(
        AxismaxSample(scalability_param=1),
        outs=['y'],
        ins=['x'],
        name='test_axismax',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
