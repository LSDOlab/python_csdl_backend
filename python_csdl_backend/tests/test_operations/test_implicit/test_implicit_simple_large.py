from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class Implicit(csdl.Model):

    def initialize(self):
        self.parameters.declare('num_nodes1')
        self.parameters.declare('num_nodes2')
        self.parameters.declare('num_nodes3')
        self.parameters.declare('nlsolver')

    def define(self):
        solver_type = self.parameters['nlsolver']

        num1 = self.parameters['num_nodes1']
        num2 = self.parameters['num_nodes2']
        num3 = self.parameters['num_nodes3']

        shp = (num1, num2, num3)
        quadratic = csdl.Model()
        a = quadratic.declare_variable('a', shape=shp)
        b = quadratic.declare_variable('b', shape=shp)
        c = quadratic.declare_variable('c', shape=shp)
        x = quadratic.declare_variable('x', shape=shp)
        u = quadratic.declare_variable('u', shape=shp)

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
        solve_quadratic = self.create_implicit_operation(quadratic)
        if solver_type == 'bracket':
            solve_quadratic.declare_state('x', residual='y', val=np.ones(shp)*0.2, bracket=(np.zeros(shp), 2.0*np.ones(shp)))
            solve_quadratic.declare_state('u', residual='v', val=np.ones(shp)*0.3, bracket=(np.zeros(shp), 2.0*np.ones(shp)))
        else:
            solve_quadratic.declare_state('x', residual='y', val=np.ones(shp)*0.2)
            solve_quadratic.declare_state('u', residual='v', val=np.ones(shp)*0.3)
            if solver_type == 'newton':
                solve_quadratic.nonlinear_solver = csdl.NewtonSolver(solve_subsystems=False)
            elif solver_type == 'nlbgs':
                solve_quadratic.nonlinear_solver = csdl.NonlinearBlockGS(maxiter = 100)
            else:
                raise ValueError(f'solver type {solver_type} is unknown.')
            solve_quadratic.linear_solver = csdl.ScipyKrylov()

        d = self.create_input('d', val=1.5)

        a = self.register_output('a', csdl.expand(d/2.0, shp))
        # a = self.declare_variable('a', val=np.ones(shp)*1.5)
        b = self.declare_variable('b', val=np.ones(shp)*2.0)
        c = self.declare_variable('c', val=-np.ones(shp)*1.0)

        x, u = solve_quadratic(a, b, c)

        self.register_output('f', x[0, 0, 0]*3.0 + 0.5*a[0, 0, 0])


def test_implicit_simple_large_newton():
    nn = 3
    shp = (nn, nn, nn)
    vals_dict = {
        'u': np.ones(shp)*0.23013859,
        'f': np.array([[[1.66650262]]]),
        'x': np.ones(shp)*0.43050087,
    }
    totals_dict = {}
    run_test(
        Implicit(num_nodes1=nn, num_nodes2=nn, num_nodes3=nn, nlsolver='newton'), 
        outs = ['u', 'f',  'x'], 
        ins = ['d', 'b', 'c'],
        name = 'test_implicit_simple_large_newton',
        vals_dict=vals_dict,
        totals_dict=totals_dict,
        )


def test_implicit_simple_large_bracketed():
    nn = 3
    shp = (nn, nn, nn)
    vals_dict = {
        'u': np.ones(shp)*0.23013859,
        'f': np.array([[[1.66650262]]]),
        'x': np.ones(shp)*0.43050087,
    }
    totals_dict = {}
    run_test(
        Implicit(num_nodes1=nn, num_nodes2=nn, num_nodes3=nn, nlsolver='bracket'), 
        outs = ['u', 'f',  'x'], 
        ins = ['d', 'b', 'c'],
        name = 'test_implicit_simple_large_bracketed',
        vals_dict=vals_dict,
        totals_dict=totals_dict,
        )

def test_implicit_simple_large_nlbgs():
    nn = 3
    shp = (nn, nn, nn)
    vals_dict = {
        'u': np.ones(shp)*0.23013859,
        'f': np.array([[[1.66650262]]]),
        'x': np.ones(shp)*0.43050087,
    }
    totals_dict = {}
    run_test(
        Implicit(num_nodes1=nn, num_nodes2=nn, num_nodes3=nn, nlsolver='nlbgs'), 
        outs = ['u', 'f',  'x'], 
        ins = ['d', 'b', 'c'],
        name = 'test_implicit_simple_large_bracketed',
        vals_dict=vals_dict,
        totals_dict=totals_dict,
        )

if __name__ == '__main__':
    nn = 3
    import csdl
    model = Implicit(num_nodes1=nn, num_nodes2=nn, num_nodes3=nn, nlsolver='bracket')

    import python_csdl_backend
    sim = python_csdl_backend.Simulator(model, checkpoints=1, save_vars='all',)
    sim.run()
    rep = csdl.GraphRepresentation(model)
    rep.visualize_graph()

    print(sim['u'])
    print(sim['f'])
    print(sim['x'])

    # sim.check_totals(of = ['u', 'f',  'x'], wrt = ['d', 'b', 'c'])
    sim.check_partials()


#     from csdl_om import Simulator as OmSimulator
#     from python_csdl_backend import Simulator as LiteSimulator
#     import pytest
#     import numpy as np
#     import time
#     import matplotlib.pyplot as plt

#     def main(model_class, outs, ins, num):
#         # CSDL OM

#         model = model_class(num_nodes1=num, num_nodes2=1, num_nodes3=1)
#         sim_om = OmSimulator(model, mode='rev')
#         # sim_om.visualize_implementation()

#         start = time.time()
#         sim_om.run()
#         to = time.time() - start

#         start = time.time()
#         x1 = sim_om.prob.compute_totals(outs, ins)
#         tot = time.time() - start

#         # CSDL LITE
#         sim_lite = LiteSimulator(model)
#         sim_lite.eval_instructions.save()
#         # sim_lite.visualize_implementation()

#         # import cProfile
#         # profiler = cProfile.Profile()
#         # profiler.enable()

#         start = time.time()
#         sim_lite.run()
#         tl = time.time() - start

#         sim_lite.generate_totals(outs, ins)
#         start = time.time()
#         x = sim_lite.compute_totals(outs, ins)
#         tlt = time.time() - start

#         for key in x:
#             print(key)
#             print(np.linalg.norm(x[key].A))
#             print(np.linalg.norm(x1[key]))

#         # profiler.disable()
#         # profiler.dump_stats('output')
#         return tl, to, tlt, tot

#     # num_vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#     # num_vec = [1, 2, 3, 4, 5, 6, 7]
#     num_vec = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 60, 100, 200, 400,  600,  1000,  2000]
#     num_vec = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 60, 100, 200]

#     tl_vec = []
#     to_vec = []
#     p_vec = []

#     tlt_vec = []
#     tot_vec = []
#     pt_vec = []
#     for num in num_vec:
#         tl, to, tlt, tot = main(Implicit, ['f'], ['a', 'b', 'c'], num)
#         tl_vec.append(tl)
#         to_vec.append(to)
#         p_vec.append(to/tl)

#         tlt_vec.append(tlt)
#         tot_vec.append(tot)
#         pt_vec.append(tot/tlt)
#         print('================', num, '================')

#     plt.figure()
#     plt.loglog(num_vec, tl_vec)
#     plt.loglog(num_vec, to_vec)
#     plt.grid()
#     plt.xlabel('size')
#     plt.ylabel('run time(sec)')

#     plt.figure()
#     plt.loglog(num_vec, p_vec)
#     plt.grid()
#     plt.xlabel('size')
#     plt.ylabel('run (om time)/(lite time)')

#     plt.figure()
#     plt.loglog(num_vec, tlt_vec)
#     plt.loglog(num_vec, tot_vec)
#     plt.grid()
#     plt.xlabel('size')
#     plt.ylabel('total time(sec)')

#     plt.figure()
#     plt.loglog(num_vec, pt_vec)
#     plt.grid()
#     plt.xlabel('size')
#     plt.ylabel('total (om time)/(lite time)')

#     plt.show()
