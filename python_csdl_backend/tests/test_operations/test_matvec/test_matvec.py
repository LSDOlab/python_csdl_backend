from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class MatvecSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        in_val1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        in_val2 = np.array([2.0, 1.0])

        x = self.create_input('x', val=in_val1)
        w = self.create_input('w', val=in_val2)

        y = csdl.matvec(x, w)

        self.register_output('y', y)


def test_matvec():
    nn = (2, 2)
    in_val1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    in_val2 = np.array([2.0, 1.0])
    val = in_val1.dot(in_val2)
    vals_dict = {'y': val}

    totalx = np.array([[2., 1., 0., 0.], [0., 0., 2., 1.]])
    totalw = in_val1
    totals_dict = {
        ('y', 'x'): totalx,
        ('y', 'w'): totalw,
    }

    run_test(
        MatvecSample(scalability_param=nn),
        outs=['y'],
        ins=['x', 'w'],
        name='test_matvec',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


class MatvecSample2(csdl.Model):

    def initialize(self):
        self.parameters.declare('val_1')
        self.parameters.declare('val_2')

    def define(self):
        v1 = self.parameters['val_1']
        v2 = self.parameters['val_2']

        x = self.create_input('x', val=v1)
        w = self.create_input('w', val=v2)

        y = csdl.matvec(x, w)+1

        self.register_output('y', y)


def test_matvec2():
    np.random.seed(10)
    in_val1 = np.random.rand(20, 20)
    in_val2 = np.random.rand(20,)
    val = in_val1 @ in_val2 + 1
    vals_dict = {'y': val}

    totals_dict = {}

    run_test(
        MatvecSample2(val_1=in_val1, val_2=in_val2),
        outs=['y'],
        ins=['x', 'w'],
        name='test_matvec2',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


class MatvecSampleSparse(csdl.Model):

    def initialize(self):
        self.parameters.declare('val_1')
        self.parameters.declare('A')

    def define(self):
        v1 = self.parameters['val_1']
        A = self.parameters['A']
        self.A = A

        x = self.create_input('x', val=v1)
        y = csdl.matvec(A, x)+1

        self.register_output('y', y)


def test_matvec_sparse():
    np.random.seed(10)
    in_val1 = np.random.rand(100)
    in_val2 = np.random.rand(100)
    import scipy.sparse as sp
    A = sp.diags(in_val2)

    val = A @ in_val1 + 1
    vals_dict = {'y': val}

    totals_dict = {('y', 'x'):A.toarray()}

    run_test(
        MatvecSampleSparse(val_1=in_val1, A=A),
        outs=['y'],
        ins=['x', 'w'],
        name='test_matvec2',
        vals_dict=vals_dict,
        totals_dict=totals_dict)

if __name__ == '__main__':
    # test_matvec()
    test_matvec2()
    # test_matvec_sparse()