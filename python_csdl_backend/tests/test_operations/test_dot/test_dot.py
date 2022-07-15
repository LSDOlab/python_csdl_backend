from csdl import Model
from csdl_om import Simulator
from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class DotSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        w = self.create_input('w', val=3*np.ones(scalability_param))
        x = self.create_input('x', val=4*np.ones(scalability_param))

        y = csdl.dot(w, x)

        self.register_output('y', y)


def test_dot():
    nn = 3
    val = np.dot(3*np.ones(3), 4*np.ones(3))
    vals_dict = {'y': val}

    totals_dict = {
        ('y', 'x'): 3*np.ones((1, 3)),
        ('y', 'w'): 4*np.ones((1, 3)),
    }

    run_test(
        DotSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_dot',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


class TensorDotSample(csdl.Model):

    def define(self):
        m = 3
        n = 4
        p = 5

        # Shape of the tensors
        ten_shape = (m, n, p)

        # Number of elements in the tensors
        num_ten_elements = np.prod(ten_shape)

        # Values for the two tensors
        ten1 = np.arange(num_ten_elements).reshape(ten_shape)
        ten2 = np.arange(num_ten_elements,
                         2 * num_ten_elements).reshape(ten_shape)

        # Adding the tensors to csdl
        ten1 = self.create_input('x', val=ten1)
        ten2 = self.create_input('w', val=ten2)

        # Tensor-Tensor Dot Product specifying the first axis
        self.register_output('y',
                             csdl.dot(ten1, ten2, axis=0))


def test_tensor_dot():
    m = 3
    n = 4
    p = 5

    # Shape of the tensors
    ten_shape = (m, n, p)

    # Number of elements in the tensors
    num_ten_elements = np.prod(ten_shape)

    ten1 = np.arange(num_ten_elements).reshape(ten_shape)
    ten2 = np.arange(num_ten_elements, 2 * num_ten_elements).reshape(ten_shape)

    vals_dict = {'y': np.sum(ten1 * ten2, axis=0)}

    totals_dict = {}

    run_test(
        TensorDotSample(),
        outs=['y'],
        ins=['x', 'w'],
        name='test_tensor_dot',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
