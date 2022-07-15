from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class OuterSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        x = self.create_input('x', val=0.5*np.ones(scalability_param))
        w = self.create_input('w', val=1.5*np.ones(scalability_param))

        y = csdl.outer(x, w)

        self.register_output('y', y)


def test_outer():
    nn = 1
    in_val1 = 0.5*np.ones(nn)
    in_val2 = 1.5*np.ones(nn)
    val = np.outer(in_val1, in_val2)
    vals_dict = {'y': val}

    totals_dict = {}

    run_test(
        OuterSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_outer',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


class OuterTensorVector(csdl.Model):
    def define(self):

        m = 3
        n = 4
        p = 5

        # Shape of the vectors
        vec_shape = (m, )

        # Shape of the tensors
        ten_shape = (m, n, p)

        # Values for the two vectors
        vec1 = np.arange(m)

        # Number of elements in the tensors
        num_ten_elements = np.prod(ten_shape)

        # Values for the two tensors
        ten1 = np.arange(num_ten_elements).reshape(ten_shape)

        # Adding the vector and tensor to csdl
        vec1 = self.create_input('vec1', val=vec1)

        ten1 = self.create_input('ten1', val=ten1)

        # Tensor-Vector Outer Product specifying the first axis for Vector and Tensor
        self.register_output('TenVecOuter', csdl.outer(ten1, vec1))


def test_outer_large():
    nn = (10, 10)
    m = 3
    n = 4
    p = 5

    # Shape of the vectors
    vec_shape = (m, )

    # Shape of the tensors
    ten_shape = (m, n, p)

    # Values for the two vectors
    vec1 = np.arange(m)

    # Number of elements in the tensors
    num_ten_elements = np.prod(ten_shape)

    # Values for the two tensors
    ten1 = np.arange(num_ten_elements).reshape(ten_shape)

    vals_dict = {'TenVecOuter': np.einsum('ijk,l->ijkl', ten1, vec1)}

    totals_dict = {}

    run_test(
        OuterTensorVector(),
        outs=['TenVecOuter'],
        ins=['vec1', 'ten1'],
        name='test_outer_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
