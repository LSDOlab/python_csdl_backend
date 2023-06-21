from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np

class Implicit(csdl.Model):
    def initialize(self):
        self.parameters.declare('nlsolver')

    def define(self):

        solver_type = self.parameters['nlsolver']
        quadratic = csdl.Model()
        a = quadratic.declare_variable('a')
        b = quadratic.declare_variable('b')
        c = quadratic.declare_variable('c')
        x = quadratic.declare_variable('x')

        # test_var = x**2
        # quadratic.register_output('test_var', test_var*2.0)
        # temp = quadratic.declare_variable('temp')

        # quadratic.connect(test_var.name, 'temp')
        # ax2 = a*temp
        # quadratic.register_output('t', a*1.0)
        ax2 = a*x**2
        quadratic.register_output('t', a*1.0)
        quadratic.register_output('t2', x*3.0)

        y = x - (-ax2 - c)/b

        quadratic.register_output('y', y)
        quadratic.register_output('ax2', a*x*x)

        # from csdl_om import Simulator
        # sim = Simulator(quadratic)
        # sim.visualize_implementation()
        # exit()

        solve_quadratic = self.create_implicit_operation(quadratic)
        if solver_type == 'bracket':
            solve_quadratic.declare_state('x', residual='y', val=0.34, bracket=(0.0, 4.0))
        else:
            solve_quadratic.declare_state('x', residual='y', val=0.34)

            if solver_type == 'newton':
                solve_quadratic.nonlinear_solver = csdl.NewtonSolver(solve_subsystems=False)
            elif solver_type == 'nlbgs':
                solve_quadratic.nonlinear_solver = csdl.NonlinearBlockGS(maxiter = 100)
            else:
                raise ValueError(f'solver type {solver_type} is unknown.')

        solve_quadratic.linear_solver = csdl.ScipyKrylov()

        a = self.create_input('a', val=1.5)
        b = self.create_input('b', val=2.0)
        c = self.create_input('c', val=-1.0)
        x, ax2 = solve_quadratic(a, b, c, expose=['ax2'])

        self.register_output('ax2_out', ax2*1.0)


def test_implicit_newton():
    vals_dict = {'x': np.array([0.38742589])}
    totals_dict = {}
    run_test(
        Implicit(nlsolver='newton'), 
        ['ax2_out', 'x'], 
        ['a', 'b', 'c'],
        vals_dict=vals_dict,
        totals_dict=totals_dict,
    )


def test_implicit_brackey():
    vals_dict = {'x': np.array([0.38742589])}
    totals_dict = {}
    run_test(
        Implicit(nlsolver='bracket'), 
        ['ax2_out', 'x'], 
        ['a', 'b', 'c'],
        vals_dict=vals_dict,
        totals_dict=totals_dict
    )

def test_implicit_nlbgs():
    vals_dict = {'x': np.array([0.38742589])}
    totals_dict = {}
    run_test(
        Implicit(nlsolver='nlbgs'), 
        ['ax2_out', 'x'], 
        ['a', 'b', 'c'],
        vals_dict=vals_dict,
        totals_dict=totals_dict
    )

if __name__ == '__main__':
    import python_csdl_backend
    sim = python_csdl_backend.Simulator(Implicit(nlsolver='nlbgs'))
    # sim.eval_instructions.script.save()
    # sim.run()
    # exit()
    test_implicit_nlbgs()

    # a 1.5
    # b 2.0
    # c 1.0
    # a 1.5
    # x 1.0
    # c 1.0
    # b 2.0

    # (of,wrt)             calc norm                relative error             absolute error
    # -----------------------------------------------------------------------------------------------
    # ('x', 'a')           0.047465413795168304     2.0527621568725488e-07     9.743518519789163e-09
    # ('x', 'b')           0.12251482266864411      2.695348075166658e-07      3.3022000245352956e-08
    # ('x', 'c')           0.31622776603004066      1.4481687737703644e-07     4.579512424829346e-08
    # ('ax2', 'a')         0.09493082759033661      2.3275965090994497e-07     2.2096061147447443e-08
    # ('ax2', 'b')         0.1423962414110803       4.276623636721927e-07      6.089748713633547e-08
    # ('ax2', 'c')         0.36754446793991863      2.6331389782070186e-07     9.67795409922978e-08
    # ('ax2_out', 'a')     0.09493082759033661      2.3275965090994497e-07     2.2096061147447443e-08
    # ('ax2_out', 'b')     0.1423962414110803       4.276623636721927e-07      6.089748713633547e-08
    # ('ax2_out', 'c')     0.36754446793991863      2.6331389782070186e-07     9.67795409922978e-08
    # -----------------------------------------------------------------------------------------------


