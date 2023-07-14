def get_model():
    from python_csdl_backend.tests.test_structures.utils import build_braid
    import numpy as np 
    import csdl
    from csdl import Model, CustomExplicitOperation


    class Example(CustomExplicitOperation):
        def initialize(self):
            self.parameters.declare('name1')
            self.parameters.declare('out1')
            self.parameters.declare('shape1')
            self.parameters.declare('name2')
            self.parameters.declare('out2')

        def define(self):
            self.name1 = self.parameters['name1']
            self.name2 = self.parameters['name2']
            self.out1 = self.parameters['out1']
            self.out2 = self.parameters['out2']
            self.shape = self.parameters['shape1']

            self.add_input(self.name1, shape=self.shape)
            self.add_input(self.name2, shape=self.shape)

            self.add_output(self.out1, shape=self.shape)
            self.add_output(self.out2, shape=self.shape)

            self.declare_derivatives('*', '*')

        def compute(self, inputs, outputs):
            outputs[self.out1] = inputs[self.name1]**2 +  inputs[self.name2]**2 
            outputs[self.out2] = inputs[self.name1]*inputs[self.name2] 

        def compute_derivatives(self, inputs, derivatives):
            derivatives[self.out1, self.name1] = np.diag(2*inputs[self.name1])
            derivatives[self.out1, self.name2] = np.diag(2*inputs[self.name2])
            derivatives[self.out2, self.name1] = np.diag(inputs[self.name2])
            derivatives[self.out2, self.name2] = np.diag(inputs[self.name1])


    class Implicit(csdl.Model):
        def initialize(self):
            self.parameters.declare('nlsolver')
            self.parameters.declare('in_name1')
            self.parameters.declare('in_shape1')
            self.parameters.declare('in_name2')
            self.parameters.declare('in_shape2')
            self.parameters.declare('state_name')
            self.parameters.declare('state_shape')
            self.parameters.declare('exposed_name')

        def define(self):

            solver_type = self.parameters['nlsolver']
            in_name1 = self.parameters['in_name1']
            in_name2 = self.parameters['in_name2']
            in_shape1 = self.parameters['in_shape1']
            in_shape2 = self.parameters['in_shape2']
            state_name = self.parameters['state_name']
            state_shape = self.parameters['state_shape']
            exposed_name = self.parameters['exposed_name']

            quadratic = csdl.Model()
            a = quadratic.declare_variable(in_name1, shape = in_shape1)
            b = quadratic.declare_variable(in_name2, shape = in_shape2)
            x = quadratic.declare_variable(state_name, shape = state_shape)

            y = (x - x**2 - a/20)+b/2

            quadratic.register_output(exposed_name, b*x)
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
            b = self.declare_variable(in_name2, val=np.ones(in_shape2))
            x, exposed = solve_quadratic(a, b, expose=[exposed_name])

    def tito_op_1(x, y, model):
        z = x*y*0.01
        z1 = x*0.01+y*z
        return z, z1

    def tito_op_2(x, y, model):
        z = csdl.sum(x,y)/3
        return csdl.expand(z[0], shape=x.shape), csdl.expand(z[1], shape=x.shape)
    
    def tito_op_explicit(x,y, model):
        out_name_1 = x.name + y.name + 'out1'
        out_name_2 = x.name + y.name + 'out2'
        z1, z2 = csdl.custom(x, y, op=Example(name1 = x.name, name2 = y.name, out1 = out_name_1, out2 = out_name_2, shape1 = x.shape))

        return z1*1.0, z2*1.0
    
    def tito_op_implicit(x,y, model):
        out_name_1 = x.name + y.name + 'out1'
        out_name_2 = x.name + y.name + 'out2'

        im = Implicit(
            nlsolver = 'newton',
            in_name1 = x.name,
            in_shape1 = x.shape,
            in_name2 = y.name,
            in_shape2 = y.shape,
            state_name = out_name_1,
            state_shape = y.shape,
            exposed_name = out_name_2,
        )
        model.add(im)
        y = model.declare_variable(out_name_2, shape = y.shape)
        z = model.declare_variable(out_name_1, shape = y.shape)
        
        return z*1.0, y*1.0

    tito_op_list_1 = [tito_op_1,        tito_op_2, tito_op_explicit, tito_op_explicit,  tito_op_implicit, tito_op_2,        tito_op_1]
    tito_op_list_2 = [tito_op_implicit, tito_op_2, tito_op_explicit, tito_op_1,         tito_op_2,        tito_op_1,        tito_op_explicit]

    Ladder, outputs, inputs = build_braid(tito_op_list_1 = tito_op_list_1, tito_op_list_2 = tito_op_list_2, out_stride = 2)

    return Ladder(), outputs, inputs

# EVERYTHING ABOVE NEEDS EDITING
from python_csdl_backend.tests.create_single_test import run_test

def test_braid_mixed():
    import numpy as np

    vals_dict = {
            'x_1_o_a': np.array([0.021, 0.021]),
            'x_1_o_b': np.array([0.0541, 0.0541]),
            'x_1_p_a': np.array([-0.83229126, -0.83229126]),
            'x_1_p_b': np.array([-2.5801029, -2.5801029]),
            'x_3_o_a': np.array([0.78209824, 0.78209824]),
            'x_3_o_b': np.array([0.22770268, 0.22770268]),
            'x_3_p_a': np.array([0.78209824, 0.78209824]),
            'x_3_p_b': np.array([0.22770268, 0.22770268]),
            'x_5_o_a': np.array([0.0651535, 0.0651535]),
            'x_5_o_b': np.array([3.37811191e-05, 3.37811191e-05]),
            'x_5_p_a': np.array([0.20469092, 0.20469092]),
            'x_5_p_b': np.array([0.20469092, 0.20469092]),
            'x_7_o_a': np.array([6.21963339e-11, 6.21963339e-11]),
            'x_7_o_b': np.array([0.00089948, 0.00089948]),
            'x_7_p_a': np.array([0.00809067, 0.00809067]),
            'x_7_p_b': np.array([3.16585906e-08, 3.16585906e-08]),
            'final_sum': np.array([0.00899018, 0.00899018]),
    }
    
    totals_dict = {
            ('final_sum', 'x_0_o_a'): np.array([[-0.00075953, -0.00075738],
    [-0.00075953, -0.00075738]]),
            ('final_sum', 'x_0_o_b'): np.array([[-0.00080126, -0.00079916],
    [-0.00080126, -0.00079916]]),
            ('final_sum', 'x_0_p_a'): np.array([[-0.00056419, -0.00056287],
    [-0.00056419, -0.00056287]]),
            ('final_sum', 'x_0_p_b'): np.array([[0.01839212, 0.01835609],
    [0.01839212, 0.01835609]]),
            ('x_1_o_a', 'x_0_o_a'): np.array([[0.021, 0.   ],
    [0.   , 0.021]]),
            ('x_1_o_a', 'x_0_o_b'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_1_o_a', 'x_0_p_a'): np.array([[0.01, 0.  ],
    [0.  , 0.01]]),
            ('x_1_o_a', 'x_0_p_b'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_1_o_b', 'x_0_o_a'): np.array([[0.0541, 0.    ],
    [0.    , 0.0541]]),
            ('x_1_o_b', 'x_0_o_b'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_1_o_b', 'x_0_p_a'): np.array([[0.042, 0.   ],
    [0.   , 0.042]]),
            ('x_1_o_b', 'x_0_p_b'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_1_p_a', 'x_0_o_a'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_1_p_a', 'x_0_o_b'): np.array([[0.01876467, 0.        ],
    [0.        , 0.01876467]]),
            ('x_1_p_a', 'x_0_p_a'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_1_p_a', 'x_0_p_b'): np.array([[-0.18764666,  0.        ],
    [ 0.        , -0.18764666]]),
            ('x_1_p_b', 'x_0_o_a'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_1_p_b', 'x_0_o_b'): np.array([[0.05817046, 0.        ],
    [0.        , 0.05817046]]),
            ('x_1_p_b', 'x_0_p_a'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_1_p_b', 'x_0_p_b'): np.array([[-1.41399589,  0.        ],
    [ 0.        , -1.41399589]]),
            ('x_3_o_a', 'x_0_o_a'): np.array([[-0.03415419,  0.        ],
    [-0.03415419,  0.        ]]),
            ('x_3_o_a', 'x_0_o_b'): np.array([[-0.03603608,  0.        ],
    [-0.03603608,  0.        ]]),
            ('x_3_o_a', 'x_0_p_a'): np.array([[-0.0253789,  0.       ],
    [-0.0253789,  0.       ]]),
            ('x_3_o_a', 'x_0_p_b'): np.array([[0.82755418, 0.        ],
    [0.82755418, 0.        ]]),
            ('x_3_o_b', 'x_0_o_a'): np.array([[-0.01077077,  0.        ],
    [-0.01077077,  0.        ]]),
            ('x_3_o_b', 'x_0_o_b'): np.array([[-0.01051031,  0.        ],
    [-0.01051031,  0.        ]]),
            ('x_3_o_b', 'x_0_p_a'): np.array([[-0.0065927,  0.       ],
    [-0.0065927,  0.       ]]),
            ('x_3_o_b', 'x_0_p_b'): np.array([[0.18012872, 0.        ],
    [0.18012872, 0.        ]]),
            ('x_3_p_a', 'x_0_o_a'): np.array([[ 0.        , -0.03415419],
    [ 0.        , -0.03415419]]),
            ('x_3_p_a', 'x_0_o_b'): np.array([[ 0.        , -0.03603608],
    [ 0.        , -0.03603608]]),
            ('x_3_p_a', 'x_0_p_a'): np.array([[ 0.       , -0.0253789],
    [ 0.       , -0.0253789]]),
            ('x_3_p_a', 'x_0_p_b'): np.array([[0.        , 0.82755418],
    [0.        , 0.82755418]]),
            ('x_3_p_b', 'x_0_o_a'): np.array([[ 0.        , -0.01077077],
    [ 0.        , -0.01077077]]),
            ('x_3_p_b', 'x_0_o_b'): np.array([[ 0.        , -0.01051031],
    [ 0.        , -0.01051031]]),
            ('x_3_p_b', 'x_0_p_a'): np.array([[ 0.       , -0.0065927],
    [ 0.       , -0.0065927]]),
            ('x_3_p_b', 'x_0_p_b'): np.array([[0.        , 0.18012872],
    [0.        , 0.18012872]]),
            ('x_5_o_a', 'x_0_o_a'): np.array([[-0.00305732, -0.00305732],
    [-0.00305732, -0.00305732]]),
            ('x_5_o_a', 'x_0_o_b'): np.array([[-0.0032269, -0.0032269],
    [-0.0032269, -0.0032269]]),
            ('x_5_o_a', 'x_0_p_a'): np.array([[-0.00227365, -0.00227365],
    [-0.00227365, -0.00227365]]),
            ('x_5_o_a', 'x_0_p_b'): np.array([[0.07418456, 0.07418456],
    [0.07418456, 0.07418456]]),
            ('x_5_o_b', 'x_0_o_a'): np.array([[-3.1830865e-06, -3.1830865e-06],
    [-3.1830865e-06, -3.1830865e-06]]),
            ('x_5_o_b', 'x_0_o_b'): np.array([[-3.23236817e-06, -3.23236817e-06],
    [-3.23236817e-06, -3.23236817e-06]]),
            ('x_5_o_b', 'x_0_p_a'): np.array([[-2.15691878e-06, -2.15691878e-06],
    [-2.15691878e-06, -2.15691878e-06]]),
            ('x_5_o_b', 'x_0_p_b'): np.array([[6.51868106e-05, 6.51868106e-05],
    [6.51868106e-05, 6.51868106e-05]]),
            ('x_5_p_a', 'x_0_o_a'): np.array([[-0.00894174, -0.0089077 ],
    [-0.00894174, -0.0089077 ]]),
            ('x_5_p_a', 'x_0_o_b'): np.array([[-0.00943144, -0.00939822],
    [-0.00943144, -0.00939822]]),
            ('x_5_p_a', 'x_0_p_a'): np.array([[-0.00663938, -0.00661854],
    [-0.00663938, -0.00661854]]),
            ('x_5_p_a', 'x_0_p_b'): np.array([[0.21637445, 0.21580515],
    [0.21637445, 0.21580515]]),
            ('x_5_p_b', 'x_0_o_a'): np.array([[-0.00894174, -0.0089077 ],
    [-0.00894174, -0.0089077 ]]),
            ('x_5_p_b', 'x_0_o_b'): np.array([[-0.00943144, -0.00939822],
    [-0.00943144, -0.00939822]]),
            ('x_5_p_b', 'x_0_p_a'): np.array([[-0.00663938, -0.00661854],
    [-0.00663938, -0.00661854]]),
            ('x_5_p_b', 'x_0_p_b'): np.array([[0.21637445, 0.21580515],
    [0.21637445, 0.21580515]]),
            ('x_7_o_a', 'x_0_o_a'): np.array([[-1.13432126e-11, -1.13250229e-11],
    [-1.13432126e-11, -1.13250229e-11]]),
            ('x_7_o_a', 'x_0_o_b'): np.array([[-1.17346985e-11, -1.17169487e-11],
    [-1.17346985e-11, -1.17169487e-11]]),
            ('x_7_o_a', 'x_0_p_a'): np.array([[-8.04299234e-12, -8.03185861e-12],
    [-8.04299234e-12, -8.03185861e-12]]),
            ('x_7_o_a', 'x_0_p_b'): np.array([[2.52736398e-10, 2.52432197e-10],
    [2.52736398e-10, 2.52432197e-10]]),
            ('x_7_o_b', 'x_0_o_a'): np.array([[-3.99968789e-05, -3.98834087e-05],
    [-3.99968789e-05, -3.98834087e-05]]),
            ('x_7_o_b', 'x_0_o_b'): np.array([[-4.21944435e-05, -4.20837172e-05],
    [-4.21944435e-05, -4.20837172e-05]]),
            ('x_7_o_b', 'x_0_p_a'): np.array([[-2.97100786e-05, -2.96406244e-05],
    [-2.97100786e-05, -2.96406244e-05]]),
            ('x_7_o_b', 'x_0_p_b'): np.array([[0.00096853, 0.00096663],
    [0.00096853, 0.00096663]]),
            ('x_7_p_a', 'x_0_o_a'): np.array([[-0.00071953, -0.00071749],
    [-0.00071953, -0.00071749]]),
            ('x_7_p_a', 'x_0_o_b'): np.array([[-0.00075906, -0.00075707],
    [-0.00075906, -0.00075707]]),
            ('x_7_p_a', 'x_0_p_a'): np.array([[-0.00053447, -0.00053322],
    [-0.00053447, -0.00053322]]),
            ('x_7_p_a', 'x_0_p_b'): np.array([[0.01742349, 0.01738936],
    [0.01742349, 0.01738936]]),
            ('x_7_p_b', 'x_0_o_a'): np.array([[-4.50206603e-09, -4.49764883e-09],
    [-4.50206603e-09, -4.49764883e-09]]),
            ('x_7_p_b', 'x_0_o_b'): np.array([[-4.63168925e-09, -4.62737887e-09],
    [-4.63168925e-09, -4.62737887e-09]]),
            ('x_7_p_b', 'x_0_p_a'): np.array([[-3.14967526e-09, -3.14697153e-09],
    [-3.14967526e-09, -3.14697153e-09]]),
            ('x_7_p_b', 'x_0_p_b'): np.array([[9.78714055e-08, 9.77975329e-08],
    [9.78714055e-08, 9.77975329e-08]]),
    }
    m, outputs, inputs = get_model()

    from csdl import GraphRepresentation

    # rep = GraphRepresentation(m)
    # rep.visualize_graph()
    # exit()

    run_test(
        m,
        outs=outputs,
        ins=inputs,
        name='test_braid_mixed',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
    

if __name__ == '__main__':
    test_braid_mixed()