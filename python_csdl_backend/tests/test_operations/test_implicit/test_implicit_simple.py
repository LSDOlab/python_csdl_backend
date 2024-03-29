from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np
import pytest

class Implicit(csdl.Model):
    def initialize(self):
        self.parameters.declare('nlsolver')
        self.parameters.declare('lsolver')
        self.parameters.declare('use_vjp', default=True)

    def define(self):

        solver_type = self.parameters['nlsolver']
        lsolver_type = self.parameters['lsolver']
        use_vjp = self.parameters['use_vjp']

        quadratic = csdl.Model()
        a = quadratic.declare_variable('a')
        b = quadratic.declare_variable('b')
        c = quadratic.declare_variable('c')
        x = quadratic.declare_variable('x')
        u = quadratic.declare_variable('u')

        # test_var = x**2
        # quadratic.register_output('test_var', test_var*2.0)
        # temp = quadratic.declare_variable('temp')

        # quadratic.connect(test_var.name, 'temp')
        # ax2 = a*temp
        # quadratic.register_output('t', a*1.0)
        ax2 = a*x**2
        au2 = a*u**2

        y = x - (-ax2 - c)/b
        v = u - (-au2 - c/2)/b

        quadratic.register_output('y', y)
        quadratic.register_output('v', v)

        # from csdl_om import Simulator
        # sim = Simulator(quadratic)
        # sim.visualize_implementation()
        # exit()

        # SOLUTION: x [0.38742589]
        # SOLUTION: u [0.66666667]

        solve_quadratic = self.create_implicit_operation(quadratic, use_vjps=use_vjp)
        if solver_type == 'bracket':
            solve_quadratic.declare_state('x', residual='y', val=0.34, bracket=(0, 0.5))
            solve_quadratic.declare_state('u', residual='v', val=0.4, bracket=(0, 1.0))
        else:
            solve_quadratic.declare_state('x', residual='y', val=0.34)
            solve_quadratic.declare_state('u', residual='v', val=0.4)
            if solver_type == 'newton':
                solve_quadratic.nonlinear_solver = csdl.NewtonSolver(solve_subsystems=False)
            elif solver_type == 'nlbgs':
                solve_quadratic.nonlinear_solver = csdl.NonlinearBlockGS(maxiter=100)
            else:
                raise ValueError(f'solver type {solver_type} is unknown.')


        if lsolver_type == 'direct':
            solve_quadratic.linear_solver = csdl.DirectSolver()
        elif lsolver_type == 'krylov':
            solve_quadratic.linear_solver = csdl.ScipyKrylov()
        elif lsolver_type == 'lbgs':
            solve_quadratic.linear_solver = csdl.LinearBlockGS()

        a = self.create_input('a', val=1.5)
        b = self.create_input('b', val=2.0)
        c = self.create_input('c', val=-1.0)
        x, u = solve_quadratic(a, b, c)

        self.register_output('f', x*3.0 + u*3.0 + 0.5*a)
        self.register_output('nl_f', x*u**2)


def test_implicit_simple_newton():
    vals_dict = {
        'x': np.array([0.38742589]),
        'u': np.array([0.21525044]),
        'f': np.array([2.55802897]),
    }
    totals_dict = {}
    run_test(
        Implicit(nlsolver='newton', lsolver = 'direct'),
        outs=['x', 'u', 'f'],
        ins=['a', 'b', 'c'],
        name='test_implicit_simple_newton',
        vals_dict=vals_dict,
        totals_dict=totals_dict,
    )


def test_implicit_simple_bracket():
    vals_dict = {
        'x': np.array([0.38742589]),
        'u': np.array([0.21525044]),
        'f': np.array([2.55802897]),
    }
    totals_dict = {}
    run_test(
        Implicit(nlsolver='bracket', lsolver = 'direct'),
        outs=['x', 'u', 'f'],
        ins=['a', 'b', 'c'],
        name='test_implicit_simple_bracket',
        vals_dict=vals_dict,
        totals_dict=totals_dict,
    )


def test_implicit_simple_nlbgs():
    vals_dict = {
        'x': np.array([0.38742589]),
        'u': np.array([0.21525044]),
        'f': np.array([2.55802897]),
    }
    totals_dict = {}
    run_test(
        Implicit(nlsolver='nlbgs', lsolver = 'direct'),
        outs=['x', 'u', 'f'],
        ins=['a', 'b', 'c'],
        name='test_implicit_simple_nlbgs',
        vals_dict=vals_dict,
        totals_dict=totals_dict,
    )

def test_implicit_simple_newton_krylov():
    vals_dict = {
        'x': np.array([0.38742589]),
        'u': np.array([0.21525044]),
        'f': np.array([2.55802897]),
    }
    totals_dict = {}
    run_test(
        Implicit(nlsolver='newton', lsolver = 'krylov'),
        outs=['x', 'u', 'f'],
        ins=['a', 'b', 'c'],
        name='test_implicit_simple_newton',
        vals_dict=vals_dict,
        totals_dict=totals_dict,
    )


def test_implicit_simple_bracket_krylov():
    vals_dict = {
        'x': np.array([0.38742589]),
        'u': np.array([0.21525044]),
        'f': np.array([2.55802897]),
    }
    totals_dict = {}
    run_test(
        Implicit(nlsolver='bracket', lsolver = 'krylov'),
        outs=['x', 'u', 'f'],
        ins=['a', 'b', 'c'],
        name='test_implicit_simple_bracket',
        vals_dict=vals_dict,
        totals_dict=totals_dict,
    )


def test_implicit_simple_nlbgs_krylov():
    vals_dict = {
        'x': np.array([0.38742589]),
        'u': np.array([0.21525044]),
        'f': np.array([2.55802897]),
    }
    totals_dict = {}
    run_test(
        Implicit(nlsolver='nlbgs', lsolver = 'krylov'),
        outs=['x', 'u', 'f'],
        ins=['a', 'b', 'c'],
        name='test_implicit_simple_nlbgs',
        vals_dict=vals_dict,
        totals_dict=totals_dict,
    )

def test_implicit_simple_bracket_vjp():
    vals_dict = {
        'x': np.array([0.38742589]),
        'u': np.array([0.21525044]),
        'f': np.array([2.55802897]),
    }
    totals_dict = {}
    run_test(
        Implicit(nlsolver='bracket', lsolver = 'direct', use_vjp=False),
        outs=['x', 'u', 'f'],
        ins=['a', 'b', 'c'],
        name='test_implicit_simple_bracket',
        vals_dict=vals_dict,
        totals_dict=totals_dict,
    )


def test_implicit_simple_nlbgs_vjp():
    vals_dict = {
        'x': np.array([0.38742589]),
        'u': np.array([0.21525044]),
        'f': np.array([2.55802897]),
    }
    totals_dict = {}
    run_test(
        Implicit(nlsolver='nlbgs', lsolver = 'direct', use_vjp=False),
        outs=['x', 'u', 'f'],
        ins=['a', 'b', 'c'],
        name='test_implicit_simple_nlbgs',
        vals_dict=vals_dict,
        totals_dict=totals_dict,
    )

def test_implicit_simple_newton_krylov_vjp():
    vals_dict = {
        'x': np.array([0.38742589]),
        'u': np.array([0.21525044]),
        'f': np.array([2.55802897]),
    }
    totals_dict = {}
    run_test(
        Implicit(nlsolver='newton', lsolver = 'krylov', use_vjp=False),
        outs=['x', 'u', 'f'],
        ins=['a', 'b', 'c'],
        name='test_implicit_simple_newton',
        vals_dict=vals_dict,
        totals_dict=totals_dict,
    )


def test_implicit_simple_lbgs_error():
    import python_csdl_backend
    r = csdl.GraphRepresentation(Implicit(nlsolver='nlbgs', lsolver = 'lbgs'))
    with pytest.raises(NotImplementedError) as excinfo:  
        python_csdl_backend.Simulator(r)  

if __name__ == '__main__':
    # test_implicit_simple_nlbgs_krylov()
    m = Implicit(nlsolver='nlbgs', lsolver = 'krylov')


    import python_csdl_backend
    sim_lite = python_csdl_backend.Simulator(m, sparsity='sparse', display_scripts=0)

    sim_lite.run()

    outs_check_vjp = ['x','f']
    # Set first cartesian basis vector to compute vjp
    of_vectors = {}
    for i, key in enumerate(outs_check_vjp):
        of_vectors[key] = np.zeros(sim_lite[key].shape).flatten()
        if i == 0:
            check_key = key
            of_vectors[key][0] = 1.0

    in_vars = [in_name for in_name in sim_lite.variable_info['leaf_start'].keys()]
    vjp_dict = sim_lite.compute_vector_jacobian_product(of_vectors=of_vectors, wrt=in_vars)

    # Lets make sure that the first row of the derivatives is equal
    check_dict = sim_lite.compute_totals(of=outs_check_vjp, wrt=in_vars)
    for key_deriv in check_dict:
        of_var = key_deriv[0]
        wrt_var = key_deriv[1]
        if of_var != check_key:
            continue

        if isinstance(check_dict[key_deriv], np.ndarray):
            check_vector = check_dict[key_deriv][0,:]
        else:
            check_vector = check_dict[key_deriv].toarray()[0,:]
        
        # print()
        print(check_vector.flatten(),vjp_dict[wrt_var].flatten())
        np.testing.assert_almost_equal(
            check_vector.flatten(),
            vjp_dict[wrt_var].flatten(),
            decimal=5)

    exit()
    # quadratic = csdl.Model()
    # a = quadratic.declare_variable('a')
    # b = quadratic.declare_variable('b')
    # c = quadratic.declare_variable('c')
    # x = quadratic.declare_variable('x')
    # u = quadratic.declare_variable('u')

    # # test_var = x**2
    # # quadratic.register_output('test_var', test_var*2.0)
    # # temp = quadratic.declare_variable('temp')

    # # quadratic.connect(test_var.name, 'temp')
    # # ax2 = a*temp
    # # quadratic.register_output('t', a*1.0)
    # ax2 = a*x**2
    # au2 = a*u**2

    # y = x - (-ax2 - c)/b
    # v = u - (-au2 - c/2)/b

    # quadratic.register_output('y', y)
    # quadratic.register_output('v', v)

    import python_csdl_backend

    m = Implicit(nlsolver='nlbgs')
    sim = python_csdl_backend.Simulator(m, sparsity='sparse', display_scripts=1)
    sim.run()
    # sim.compute_totals(of = 'f', wrt = 'a')
    sim.compute_totals(of = ['f'], wrt = 'a')
    exit()



    run_test(
        Implicit(nlsolver='nlbgs'),
        outs=['x', 'u', 'f'],
        ins=['a', 'b', 'c'],
        name='test_implicit_simple_nlbgs',
        vals_dict=set(),
        totals_dict=set(),
    )
    exit()

#     from csdl_om import Simulator as OmSimulator
#     from python_csdl_backend import Simulator as LiteSimulator
#     import pytest
#     import numpy as np
#     import time
#     import matplotlib.pyplot as plt
#     outs = ['x', 'u', 'f']
#     ins = ['a', 'b', 'c']
#     model = Implicit()

#     # outs = ['y', 'v']
#     # ins = ['x', 'u', 'a', 'b', 'c', ]
#     # quadratic = csdl.Model()
#     # a = quadratic.declare_variable('a', val=1.5)
#     # b = quadratic.declare_variable('b', val=2.0)
#     # c = quadratic.declare_variable('c', val=-1.0)
#     # x = quadratic.declare_variable('x', val=0.3)
#     # u = quadratic.declare_variable('u', val=0.4)
#     # ax2 = a*x**2
#     # au2 = a*u**2
#     # y = x - (-ax2 - c)/b
#     # v = u - (-au2 - c*2)/b
#     # quadratic.register_output('y', y)
#     # quadratic.register_output('v', v)
#     # model = quadratic

#     # outs = ['y']
#     # ins = ['x']
#     # quadratic = csdl.Model()
#     # c = quadratic.declare_variable('c', val=-1.0)
#     # x = quadratic.declare_variable('x', val=0.32)
#     # y = x+(x**2)
#     # quadratic.register_output('y', y)
#     # model = quadratic

#     sim_om = OmSimulator(model)
#     # sim_om.visualize_implementation()
#     sim_om.run()

#     start = time.time()
#     toto = sim_om.prob.compute_totals(outs, ins)
#     to = time.time() - start

#     # CSDL LITE
#     sim_lite = LiteSimulator(model)
#     sim_lite.check_partials()
#     sim_lite.eval_instructions.save()
#     # sim_lite.visualize_implementation()
#     # plt.show()
#     sim_lite.generate_totals(outs, ins, save_script=True)

#     sim_lite.run()
#     start = time.time()
#     totl = sim_lite.compute_totals(outs, ins)
#     tl = time.time() - start

#     print('\n---OLD---')

#     for out in outs:
#         print(out, sim_lite[out])
#         print(out, sim_om[out])
#     print()
#     for key in totl:
#         print(key, 'lite', totl[key])
#         print(key, ' om ', toto[key])
#     print('---OLD---\n')

# # iteration 0, y error: 0.07329999999999998
# # ('y', 'x') [[1.51]]
# # ('y', 'u')
# # ('y', 'a') [[0.0578]]
# # ('y', 'b') [[0.20665]]
# # ('y', 'c') [[0.5]]
# # ('v', 'x')
# # ('v', 'u') [[1.6]]
# # ('v', 'a') [[0.08]]
# # ('v', 'b') [[0.44]]
# # ('v', 'c') [[1.]]
# # iteration 1, y error: 0.001767320512258197
# # ('y', 'x') [[1.58281457]]
# # ('y', 'u')
# # ('y', 'a') [[0.07548285]]
# # ('y', 'b') [[0.19338786]]
# # ('y', 'c') [[0.5]]
# # ('v', 'x')
# # ('v', 'u') [[2.05]]
# # ('v', 'a') [[0.245]]
# # ('v', 'b') [[0.31625]]
# # ('v', 'c') [[1.]]
# # iteration 2, y error: 9.350435120625278e-07
# # ('y', 'x') [[1.58113972]]
# # ('y', 'u')
# # ('y', 'a') [[0.07504964]]
# # ('y', 'b') [[0.19371277]]
# # ('y', 'c') [[0.5]]
# # ('v', 'x')
# # ('v', 'u') [[2.00060976]]
# # ('v', 'a') [[0.22249331]]
# # ('v', 'b') [[0.33313002]]
# # ('v', 'c') [[1.]]
# # iteration 3, y error: 2.62234678416462e-13
# # iteration 3, v error: 1.238963956984307e-07
# # ('y', 'x') [[1.58113883]]
# # ('y', 'u')
# # ('y', 'a') [[0.07504941]]
# # ('y', 'b') [[0.19371294]]
# # ('y', 'c') [[0.5]]
# # ('v', 'x')
# # ('v', 'u') [[2.00000009]]
# # ('v', 'a') [[0.22222226]]
# # ('v', 'b') [[0.3333333]]
# # ('v', 'c') [[1.]]

# # iteration 0, y error: 0.07329999999999998
# # ('y', 'x') [[1.51]]
# # ('y', 'u') [[0.]]
# # ('y', 'a') [[0.0578]]
# # ('y', 'b') [[0.20665]]
# # ('y', 'c') [[0.5]]
# # ('v', 'x') [[0.]]
# # ('v', 'u') [[1.6]]
# # ('v', 'a') [[0.08]]
# # ('v', 'b') [[0.44]]
# # ('v', 'c') [[1.]]
# # /Users/zensperry/opt/anaconda3/lib/python3.8/site-packages/scipy/sparse/_index.py:146: SparseEfficiencyWarning: Changing the sparsity structure of a csc_matrix is expensive. lil_matrix is more efficient.
# #   self._set_arrayXarray(i, j, x)
# # iteration 1, y error: 0.001767320512258197
# # ('y', 'x') [[2.09281457]]
# # ('y', 'u') [[0.]]
# # ('y', 'a') [[0.07548285]]
# # ('y', 'b') [[0.19338786]]
# # ('y', 'c') [[0.5]]
# # ('v', 'x') [[0.]]
# # ('v', 'u') [[2.65]]
# # ('v', 'a') [[0.245]]
# # ('v', 'b') [[0.31625]]
# # ('v', 'c') [[1.]]
# # iteration 2, y error: 0.0004312148874223154
# # ('y', 'x') [[2.67436243]]
# # ('y', 'u') [[0.]]
# # ('y', 'a') [[0.07515509]]
# # ('y', 'b') [[0.19363368]]
# # ('y', 'c') [[0.5]]
# # ('v', 'x') [[0.]]
# # ('v', 'u') [[3.66179245]]
# # ('v', 'a') [[0.22749422]]
# # ('v', 'b') [[0.32937934]]
# # ('v', 'c') [[1.]]
# # iteration 3, y error: 0.0001762252014338439
# # ('y', 'x') [[3.25566844]]
# # ('y', 'u') [[0.]]
# # ('y', 'a') [[0.07509259]]
# # ('y', 'b') [[0.19368056]]
# # ('y', 'c') [[0.5]]
# # ('v', 'x') [[0.]]
# # ('v', 'u') [[4.66712511]]
# # ('v', 'a') [[0.22459861]]
# # ('v', 'b') [[0.33155104]]
# # ('v', 'c') [[1.]]
# # iteration 4, y error: 9.063331139425035e-05
# # ('y', 'x') [[3.83689325]]
# # ('y', 'u') [[0.]]
# # ^C('y', 'a') [[0.07507162]]
