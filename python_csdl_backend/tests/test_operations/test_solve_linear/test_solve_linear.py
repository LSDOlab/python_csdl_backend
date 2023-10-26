from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np
import scipy.sparse as sp


class SolveLinearSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('type')
        self.parameters.declare('solver', default=csdl.DirectSolver())

    def define(self):
        type_solve = self.parameters['type']
        solver = self.parameters['solver']

        A = self.create_input('A', val=A_val)
        b = self.create_input('b', val=b_val)

        if type_solve == 'varvar':
            self.register_output('x', csdl.solve(A, b, solver = solver))
        elif type_solve == 'npvar':
            self.register_output('x', csdl.solve(A_val, b, solver = solver))
        elif type_solve == 'spvar':
            self.register_output('x', csdl.solve(sp.csc_matrix(A_val), b, solver = solver))
        elif type_solve == 'varnp':
            self.register_output('x', csdl.solve(A, b_val, solver = solver))
        elif type_solve == 'npnp':
            self.register_output('x', csdl.solve(A_val, b_val, solver = solver))
        elif type_solve == 'error1':
            A = A.reshape(A.shape + (1,))
            self.register_output('x', csdl.solve(A, b, solver = solver))
        elif type_solve == 'error2':
            A = A.flatten()
            self.register_output('x', csdl.solve(A, b, solver = solver))
        elif type_solve == 'error3':
            b = b.reshape(b.shape + (1,1,))
            self.register_output('x', csdl.solve(A, b, solver = solver))
        elif type_solve == 'error4':
            self.register_output('x', csdl.solve(A, b, solver = 'A'))
        elif type_solve == 'error5':
            b = np.ones((size_b+1,))
            self.register_output('x', csdl.solve(A, b, solver = solver))

size_mat = 20
size_b = 1
main_diag = np.arange(size_mat)+1
A_val = np.diag(main_diag) + np.diag(main_diag[:-1]+1, 1) + np.diag(main_diag[:-1]+2, -1)
b_val = 2*np.arange(size_mat)


def test_solvelinear_varvar():
    vals_dict = {'x': np.linalg.solve(A_val, b_val)}
    totals_dict = {}

    run_test(
        SolveLinearSample(
            type = 'varvar'),
        outs=['x'],
        ins=['b', 'A'],
        vals_dict=vals_dict,
        totals_dict=totals_dict)

def test_solvelinear_npvar():
    vals_dict = {'x': np.linalg.solve(A_val, b_val)}
    totals_dict = {}

    run_test(
        SolveLinearSample(
            type = 'npvar'),
        outs=['x'],
        ins=['b', 'A'],
        vals_dict=vals_dict,
        totals_dict=totals_dict)

def test_solvelinear_spvar():
    vals_dict = {'x': np.linalg.solve(A_val, b_val)}
    totals_dict = {}

    run_test(
        SolveLinearSample(
            type = 'spvar'),
        outs=['x'],
        ins=['b', 'A'],
        vals_dict=vals_dict,
        totals_dict=totals_dict)

def test_solvelinear_varnp():
    vals_dict = {'x': np.linalg.solve(A_val, b_val)}
    totals_dict = {}

    run_test(
        SolveLinearSample(
            type = 'varnp'),
        outs=['x'],
        ins=['b', 'A'],
        vals_dict=vals_dict,
        totals_dict=totals_dict)

def test_solvelinear_npnp():
    vals_dict = {'x': np.linalg.solve(A_val, b_val)}
    totals_dict = {}
    import pytest
    import python_csdl_backend
    with pytest.raises(TypeError) as excinfo:  
        m = SolveLinearSample(type = 'npnp')
        sim = python_csdl_backend.Simulator(m)    


def test_solvelinear_varvar_krylov():
    vals_dict = {'x': np.linalg.solve(A_val, b_val)}
    totals_dict = {}

    run_test(
        SolveLinearSample(
            type = 'varvar', solver = csdl.ScipyKrylov()),
        outs=['x'],
        ins=['b', 'A'],
        vals_dict=vals_dict,
        totals_dict=totals_dict)

def test_solvelinear_npvar_krylov():
    vals_dict = {'x': np.linalg.solve(A_val, b_val)}
    totals_dict = {}

    run_test(
        SolveLinearSample(
            type = 'npvar', solver = csdl.ScipyKrylov()),
        outs=['x'],
        ins=['b', 'A'],
        vals_dict=vals_dict,
        totals_dict=totals_dict)

def test_solvelinear_spvar_krylov():
    vals_dict = {'x': np.linalg.solve(A_val, b_val)}
    totals_dict = {}

    run_test(
        SolveLinearSample(
            type = 'spvar', solver = csdl.ScipyKrylov()),
        outs=['x'],
        ins=['b', 'A'],
        vals_dict=vals_dict,
        totals_dict=totals_dict)

def test_solvelinear_varnp_krylov():
    vals_dict = {'x': np.linalg.solve(A_val, b_val)}
    totals_dict = {}

    run_test(
        SolveLinearSample(
            type = 'varnp', solver = csdl.ScipyKrylov()),
        outs=['x'],
        ins=['b', 'A'],
        vals_dict=vals_dict,
        totals_dict=totals_dict)

def test_solvelinear_arg_errors():
    vals_dict = {'x': np.linalg.solve(A_val, b_val)}
    totals_dict = {}
    import pytest
    import python_csdl_backend
    with pytest.raises(ValueError) as excinfo:  
        m = SolveLinearSample(type = 'error1')
        sim = python_csdl_backend.Simulator(m)   

    with pytest.raises(ValueError) as excinfo:  
        m = SolveLinearSample(type = 'error2')
        sim = python_csdl_backend.Simulator(m)   

    with pytest.raises(ValueError) as excinfo:  
        m = SolveLinearSample(type = 'error3')
        sim = python_csdl_backend.Simulator(m)   

    with pytest.raises(TypeError) as excinfo:  
        m = SolveLinearSample(type = 'error4')
        sim = python_csdl_backend.Simulator(m)   

    with pytest.raises(ValueError) as excinfo:  
        m = SolveLinearSample(type = 'error5')
        sim = python_csdl_backend.Simulator(m)  

if __name__ == '__main__':
        
    types = ['varvar', 'npvar', 'spvar', 'varnp']
    types = ['npvar', 'spvar', 'varnp', 'varvar']
    for type_str in types:
        sol = np.linalg.solve(A_val, b_val)


        # m = SolveLinearSample(type = type_str)
        # import python_csdl_backend

        # sim = python_csdl_backend.Simulator(m, display_scripts=0)
        # sim.run()

        # print(np.linalg.norm(sim['x'] - sol))
        # # print(sim['x'])
        # # totals = sim.compute_totals(of = ['x'], wrt = ['A', 'b'])
        # totals = sim.check_partials(compact_print=1)
        # continue
        # print(sim['x'])
        # print(totals['x', 'A'].getnnz())
        # print(totals['x', 'b'])
        
        print(sol)

        # dx/db
        vs = np.linalg.solve(A_val.T, np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).reshape(10,1))
        # vs = np.linalg.solve(A_val.T, np.eye(size_mat))
        # print('error db', np.linalg.norm(vs.T - totals['x', 'b']))

        # dx/dA
        dx_dA = -np.outer(sol,vs.T).T.reshape(1,size_mat*size_mat)
        # print('error dA', np.linalg.norm(dx_dA - totals['x', 'A']))
    #  dx/dA
    #     [[ 0.5   -0.25  -0.5    0.25 ]
    #       [-0.75   0.375  0.25  -0.125]]

    #  dx/db
    #       [[-0.5   0.5 ]
    #       [ 0.75 -0.25]]