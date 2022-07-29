from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class ElementWiseMinSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):

        m = 2
        n = 3
        # Shape of the three tensors is (2,3)
        shape = (m, n)

        # Creating the values for two tensors
        val1 = np.array([[1, 5, -8], [10, -3, -5]])
        val2 = np.array([[2, 6, 9], [-1, 2, 4]])

        # Declaring the two input tensors
        tensor1 = self.declare_variable('x', val=val1)
        tensor2 = self.declare_variable('w', val=val2)

        # Creating the output for matrix multiplication
        ma = self.register_output('y',
                                  csdl.min(tensor1, tensor2))


def test_elementwisemin():
    tensor1 = np.array([[1, 5, -8], [10, -3, -5]])
    tensor2 = np.array([[2, 6, 9], [-1, 2, 4]])

    val = np.minimum(tensor1, tensor2)
    vals_dict = {'y': val}

    # total = np.eye(nn)
    totals_dict = {}

    run_test(
        ElementWiseMinSample(scalability_param=1),
        outs=['y'],
        ins=['x', 'w'],
        name='test_elementwisemin',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
