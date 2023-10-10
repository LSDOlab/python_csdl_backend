from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np
import scipy.sparse as sp


class SparsematmatSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        in_val1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        in_val2 = np.array([[2.0, 1.0], [2.0, 4.0]])

        x = self.create_input('x', val=in_val1)
        A = sp.csc_matrix(in_val2)
        y = csdl.sparsematmat(x, A)

        self.register_output('y', y)


def test_sparsematmat():
    nn = (2, 2)
    in_val1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    in_val2 = np.array([[2.0, 1.0], [2.0, 4.0]])
    A = sp.csc_matrix(in_val2)
    val = A @ in_val1
    vals_dict = {'y': val}

    # totals = np.vstack([in_val2, in_val2])
    # totals_dict = {
    #     ('y', 'x'): totals,
    # }
    totals_dict = {}

    run_test(
        SparsematmatSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_sparsematmat',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


class SparsematmatSample2(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        in_val1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        in_val2 = np.array([[2.0, 1.0], [2.0, 4.0]])

        x = self.create_input('x', val=in_val1)
        A = sp.csc_matrix(in_val2)
        y = csdl.sparsematmat(x, A)

        self.register_output('y', y)


def test_sparsematmat2():
    nn = (2, 2)
    in_val1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    in_val2 = np.array([[2.0, 1.0], [2.0, 4.0]])
    A = sp.csc_matrix(in_val2)
    val = A @ in_val1
    vals_dict = {'y': val}

    # totals = np.vstack([in_val2, in_val2])
    # totals_dict = {
    #     ('y', 'x'): totals,
    # }
    totals_dict = {}

    run_test(
        SparsematmatSample2(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_sparsematmat',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
