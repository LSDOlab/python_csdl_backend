from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np

class MultipleTensorRandomSample(csdl.Model):
    def initialize(self):
        self.parameters.declare('axis_bool')
        self.parameters.declare('v1')
        self.parameters.declare('v2')

    def define(self):
        
        axis_bool = self.parameters['axis_bool']
        v1 = self.parameters['v1']
        v2 = self.parameters['v2']
        n = 3
        m = 6
        p = 7
        q = 10

        # Declare a tensor of shape 3x6x7x10 as input
        T1 = self.declare_variable(
            'T1',
            val=v1)

        # Declare another tensor of shape 3x6x7x10 as input
        T2 = self.declare_variable(
            'T2',
            val=v2)

        # Output the elementwise sum of tensors T1 and T2
        if axis_bool:
            self.register_output('multiple_tensor_axis_sum', csdl.sum(T1, T2, axes = (2,)))
        else:
            self.register_output('multiple_tensor_axis_sum', csdl.sum(T1, T2))

def test_sum_mult_tensor():

    n = 3
    m = 6
    p = 7
    q = 10
    np.random.seed(10)

    v1 = np.random.rand(n * m * p * q).reshape((n, m, p, q))
    v2 = np.random.rand(n * m * p * q).reshape((n, m, p, q))

    vals_dict = {'multiple_tensor_axis_sum': v1 + v2}
    totals_dict = {
        ('multiple_tensor_axis_sum', 'T1'): np.eye(n * m * p * q),
        ('multiple_tensor_axis_sum', 'T2'): np.eye(n * m * p * q),
    }

    run_test(
        MultipleTensorRandomSample(v1 = v1, v2 = v2,axis_bool = False),
        outs=['multiple_tensor_sum'],
        ins=['T1', 'T2'],
        name='test_sum_mult_axis',
        vals_dict=vals_dict,
        totals_dict=totals_dict)

def test_sum_mult_axis_tensor():

    n = 3
    m = 6
    p = 7
    q = 10
    np.random.seed(10)

    v1 = np.random.rand(n * m * p * q).reshape((n, m, p, q))
    v2 = np.random.rand(n * m * p * q).reshape((n, m, p, q))

    vals_dict = {}
    totals_dict = { }

    run_test(
        MultipleTensorRandomSample(v1 = v1, v2 = v2,axis_bool = True),
        outs=['multiple_tensor_axis_sum'],
        ins=['T1', 'T2'],
        name='test_sum_mult_axis',
        vals_dict=vals_dict,
        totals_dict=totals_dict)