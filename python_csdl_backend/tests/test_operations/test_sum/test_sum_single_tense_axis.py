from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np

class SingleTensorRandomSample(csdl.Model):
    def initialize(self):
        self.parameters.declare('v1')

    def define(self):
        
        v1 = self.parameters['v1']

        # Declare a tensor of shape 19,19 as input
        T1 = self.declare_variable(
            'T1',
            val=v1)

        self.register_output('multiple_tensor_axis_sum', csdl.sum(T1, axes = (0,1)))

def test_sum_single_tensor():

    np.random.seed(10)
    v1 = np.random.rand(19*19*19).reshape((19,19,19))

    run_test(
        SingleTensorRandomSample(v1 = v1),
        outs=['multiple_tensor_axis_sum'],
        ins=['T1'],
        name='test_sum_mult_axis',
        vals_dict={},
        totals_dict={})

