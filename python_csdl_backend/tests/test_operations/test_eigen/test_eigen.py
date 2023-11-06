from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np
import pytest

class EigenSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('matrix')

    def define(self):
        matrix = self.parameters['matrix']

        A = self.create_input('A', val=matrix)

        real_eigenvalues, imag_eigenvalues = csdl.eigenvalues(A)

        self.register_output('real', real_eigenvalues)
        self.register_output('imag', imag_eigenvalues)
        self.register_output('y',csdl.pnorm( imag_eigenvalues+real_eigenvalues**2))



def test_eigenvalue():
    nn = 20
    x = np.arange(nn)+1.0
    x1 = np.arange(nn-1)+1.0
    matrix = 1.1*np.diag(x) + 0.5*np.diag(x1, 1) + 0.2*np.diag(x1, -1)
    r, v = np.linalg.eig(matrix)

    idx = r.argsort()[::-1]
    r = r[idx]
    vals_dict = {'real': np.real(r), 'imag': np.imag(r), 'y': np.linalg.norm(np.imag(r)+np.real(r)**2)}
    totals_dict = {}
    # exit()

    run_test(
        EigenSample(matrix=matrix),
        outs=['real', 'imag', 'y'],
        ins=['A'],
        name='test_arctan',
        vals_dict=vals_dict,
        totals_dict=totals_dict)



def test_eigenvalue_eye():
    nn = 20

    matrix = 1.0*np.eye(nn)
    r, v = np.linalg.eig(matrix)

    idx = r.argsort()[::-1]
    r = r[idx]
    vals_dict = {'real': np.real(r), 'imag': np.imag(r)}
    totals_dict = {}
    # exit()

    run_test(
        EigenSample(matrix=matrix),
        outs=['real', 'imag'],
        ins=['A'],
        name='test_arctan',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_eigenvalue_eye_almost():
    nn = 15
    x = (np.arange(nn)+99)/100
    matrix = 1.0*np.diag(x)

    r, v = np.linalg.eig(matrix)

    idx = r.argsort()[::-1]
    r = r[idx]
    vals_dict = {'real': np.real(r), 'imag': np.imag(r)}
    totals_dict = {}
    # exit()

    run_test(
        EigenSample(matrix=matrix),
        outs=['real', 'imag'],
        ins=['A'],
        name='test_arctan',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_eigenvalue_rand():
    np.random.seed(42)

    nn = 15
    matrix = np.random.rand(nn,nn)
    r, v = np.linalg.eig(matrix)

    idx = r.argsort()[::-1]
    r = r[idx]
    vals_dict = {'real': np.real(r), 'imag': np.imag(r)}
    totals_dict = {}
    # exit()

    run_test(
        EigenSample(matrix=matrix),
        outs=['real', 'imag'],
        ins=['A'],
        name='test_arctan',
        vals_dict=vals_dict,
        totals_dict=totals_dict)



def test_eigenvalue_error():
    np.random.seed(42)

    nn = 15
    matrix = np.random.rand(nn,nn-1)

    with pytest.raises(ValueError) as e_info:
        rep = csdl.GraphRepresentation(EigenSample(matrix=matrix))    
        
    matrix = np.random.rand(nn)
    with pytest.raises(ValueError) as e_info:
        rep = csdl.GraphRepresentation(EigenSample(matrix=matrix))