from csdl import Model
from csdl_om import Simulator
from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class CrossSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        w = self.create_input('w', val=3*np.ones(scalability_param))
        x = self.create_input('x', val=np.array([2.0, 1.0, 0.0]))

        y = csdl.cross(w, x, axis=0)

        self.register_output('y', y)


def test_cross():
    nn = 3
    vals_dict = {'y': np.array([-3, 6, -3])}
    totals_dict = {}

    run_test(
        CrossSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_cross',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


class CrossTensorSample(Model):

    def define(self):
        # Creating two tensors
        shape = (2, 5, 4, 3)
        num_elements = np.prod(shape)

        tenval1 = np.arange(num_elements).reshape(shape)
        tenval2 = np.arange(num_elements).reshape(shape) + 6

        w = self.create_input('w', val=tenval1)
        x = self.create_input('x', val=tenval2)

        # Tensor-Tensor Dot Product specifying the last axis
        self.register_output('y', csdl.cross(w, x, axis=3))


def test_cross_large():
    shape = (2, 5, 4, 3)
    num_elements = np.prod(shape)
    w = np.arange(num_elements).reshape(shape)
    x = np.arange(num_elements).reshape(shape) + 6
    vals_dict = {'y': np.cross(w, x, axis=3)}
    totals_dict = {}

    run_test(
        CrossTensorSample(),
        outs=['y'],
        ins=['x'],
        name='test_cross_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
