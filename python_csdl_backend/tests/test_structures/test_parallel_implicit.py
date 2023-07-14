

def get_model(solver = 'newton'):
    import numpy as np 
    import csdl
    from csdl import Model
    from python_csdl_backend.tests.test_structures.utils import build_embarassing

    import csdl
    class Implicit(csdl.Model):
        def initialize(self):
            self.parameters.declare('nlsolver')
            self.parameters.declare('in_name1')
            self.parameters.declare('in_shape1')
            self.parameters.declare('state_name')
            self.parameters.declare('state_shape')

        def define(self):

            solver_type = self.parameters['nlsolver']
            in_name1 = self.parameters['in_name1']
            in_shape1 = self.parameters['in_shape1']
            state_name = self.parameters['state_name']
            state_shape = self.parameters['state_shape']

            quadratic = csdl.Model()
            a = quadratic.declare_variable(in_name1, shape = in_shape1)
            x = quadratic.declare_variable(state_name, shape = state_shape)

            # y = x - x**3 - 0.3*a
            y = x**1.1 - a
            # y = x**2 - 5*a**2

            quadratic.register_output('res', y)

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
            x = solve_quadratic(a)


    def siso_op(x, model):
        state_name = x.name + 'state'
        im = Implicit(
            nlsolver = solver,
            in_name1 = x.name,
            in_shape1 = x.shape,
            state_name = state_name,
            state_shape = x.shape,
        )
        model.add(im)
        z = model.declare_variable(state_name, shape = x.shape)
        return z*1.0

    Ladder, outputs, inputs = build_embarassing(10,3 ,siso_op = siso_op, out_stride = 5)
    # outputs = ['final_sum']
    return Ladder(), outputs, inputs

from python_csdl_backend.tests.create_single_test import run_test

def test_parallel_implicit():
    import numpy as np

    vals_dict = {
            'final_sum': np.array([[3.9352107],
    [3.9352107]]),
            'x_5_0': np.array([[1.],
    [1.]]),
            'x_9_0': np.array([[1.],
    [1.]]),
            'x_5_1': np.array([[1.53785696],
    [1.53785696]]),
            'x_9_1': np.array([[1.34173301],
    [1.34173301]]),
            'x_5_2': np.array([[1.97812969],
    [1.97812969]]),
            'x_9_2': np.array([[1.59347769],
    [1.59347769]]),
    }

    totals_dict = {
            ('final_sum', 'x_0_0'): np.array([[ 0.42409762, -0.        ],
    [-0.        ,  0.42409762]]),
            ('final_sum', 'x_0_1'): np.array([[ 0.28451289, -0.        ],
    [-0.        ,  0.28451289]]),
            ('final_sum', 'x_0_2'): np.array([[ 0.22526336, -0.        ],
    [-0.        ,  0.22526336]]),
            ('x_5_0', 'x_0_0'): np.array([[ 0.62092132, -0.        ],
    [-0.        ,  0.62092132]]),
            ('x_5_0', 'x_0_1'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_5_0', 'x_0_2'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_5_1', 'x_0_0'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_5_1', 'x_0_1'): np.array([[ 0.47744409, -0.        ],
    [-0.        ,  0.47744409]]),
            ('x_5_1', 'x_0_2'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_5_2', 'x_0_0'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_5_2', 'x_0_1'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_5_2', 'x_0_2'): np.array([[ 0.40942097, -0.        ],
    [-0.        ,  0.40942097]]),
            ('x_9_0', 'x_0_0'): np.array([[ 0.42409762, -0.        ],
    [-0.        ,  0.42409762]]),
            ('x_9_0', 'x_0_1'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_9_0', 'x_0_2'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_9_1', 'x_0_0'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_9_1', 'x_0_1'): np.array([[ 0.28451289, -0.        ],
    [-0.        ,  0.28451289]]),
            ('x_9_1', 'x_0_2'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_9_2', 'x_0_0'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_9_2', 'x_0_1'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_9_2', 'x_0_2'): np.array([[ 0.22526336, -0.        ],
    [-0.        ,  0.22526336]]),
    }

    m, outputs, inputs = get_model()

    run_test(
        m,
        outs=outputs,
        ins=inputs,
        name='test_parallel_implicit',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
    
if __name__ == '__main__':
    test_parallel_implicit()