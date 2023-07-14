def get_model():
    from python_csdl_backend.tests.test_structures.utils import build_ladder
    import numpy as np 
    import csdl
    from csdl import Model
    class Implicit(csdl.Model):
        def initialize(self):
            self.parameters.declare('nlsolver')
            self.parameters.declare('in_name1')
            self.parameters.declare('in_shape1')
            self.parameters.declare('in_name2')
            self.parameters.declare('in_shape2')
            self.parameters.declare('state_name')
            self.parameters.declare('state_shape')

        def define(self):

            solver_type = self.parameters['nlsolver']
            in_name1 = self.parameters['in_name1']
            in_name2 = self.parameters['in_name2']
            in_shape1 = self.parameters['in_shape1']
            in_shape2 = self.parameters['in_shape2']
            state_name = self.parameters['state_name']
            state_shape = self.parameters['state_shape']

            quadratic = csdl.Model()
            a = quadratic.declare_variable(in_name1, shape = in_shape1)
            b = quadratic.declare_variable(in_name2, shape = in_shape2)
            x = quadratic.declare_variable(state_name, shape = state_shape)

            # ax2 = a*x**2
            # y = x - (-ax2 - a)/b
            y = (x - x**2 - a)+b

            quadratic.register_output('res', y)
            quadratic.print_var(x)
            # self.print_var(b*1)
            # self.print_var(x*1)
            # self.print_var(y*1)


            solve_quadratic = self.create_implicit_operation(quadratic)
            if solver_type == 'bracket':
                solve_quadratic.declare_state(state_name, residual='res', val=0.34, bracket=(0.0, 4.0))
            else:
                solve_quadratic.declare_state(state_name, residual='res', val=0.34)

                if solver_type == 'newton':
                    solve_quadratic.nonlinear_solver = csdl.NewtonSolver(solve_subsystems=False)
                elif solver_type == 'nlbgs':
                    solve_quadratic.nonlinear_solver = csdl.NonlinearBlockGS(maxiter = 100)
                else:
                    raise ValueError(f'solver type {solver_type} is unknown.')

            solve_quadratic.linear_solver = csdl.ScipyKrylov()

            a = self.declare_variable(in_name1, val=np.ones(in_shape1))
            b = self.declare_variable(in_name2, val=np.ones(in_shape2))
            x = solve_quadratic(a, b)

    def tiso_op(x, y, model):
        state_name = x.name + y.name + 'state'
        im = Implicit(
            nlsolver = 'newton',
            in_name1 = x.name,
            in_shape1 = x.shape,
            in_name2 = y.name,
            in_shape2 = y.shape,
            state_name = state_name,
            state_shape = y.shape,
        )
        model.add(im)
        z = model.declare_variable(state_name, shape = y.shape)
        return z*1.0

    Ladder, outputs, inputs = build_ladder(10, siso_op = csdl.sin, tiso_op = tiso_op, out_stride = 5)

    return Ladder(), outputs, inputs

# EVERYTHING ABOVE NEEDS EDITING
from python_csdl_backend.tests.create_single_test import run_test

def test_ladder_implicit():
    import numpy as np

    vals_dict = {
            'x_5_o': np.array([[-0.81372784],
                [-0.81372784]]),
            'x_5_p': np.array([[0.59371457],
                [0.59371457]]),
            'x_9_o': np.array([[-0.72428861],
                [-0.72428861]]),
    }

    totals_dict = {
            ('x_5_o', 'x_0_o'): np.array([[ 0.00844726, -0.        ],
                [-0.        ,  0.00844726]]),
            ('x_5_o', 'x_0_p'): np.array([[0.12229056, 0.        ],
                [0.        , 0.12229056]]),
            ('x_5_p', 'x_0_o'): np.array([[0., 0.],
                [0., 0.]]),
            ('x_5_p', 'x_0_p'): np.array([[-0.14776063, -0.        ],
                [-0.        , -0.14776063]]),
            ('x_9_o', 'x_0_o'): np.array([[ 0.0002133, -0.       ],
                [-0.       ,  0.0002133]]),
            ('x_9_o', 'x_0_p'): np.array([[0.06839979, 0.        ],
                [0.        , 0.06839979]]),
    }

    m, outputs, inputs = get_model()

    run_test(
        m,
        outs=outputs,
        ins=inputs,
        name='test_ladder_implicit',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
    

if __name__ == '__main__':
    test_ladder_implicit()