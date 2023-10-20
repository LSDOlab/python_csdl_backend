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

    def tito_op_1(x, y, model):
        z = x*y*0.01
        z1 = x*0.01+y*z
        return z, z1
    
    def tito_op_explicit(x,y, model):
        out_name_1 = x.name + y.name + 'out1'
        out_name_2 = x.name + y.name + 'out2'
        z1, z2 = csdl.custom(x, y, op=Example(name1 = x.name, name2 = y.name, out1 = out_name_1, out2 = out_name_2, shape1 = x.shape))

        return z1*1.0, z2*1.0

    tito_op_list_1 = [tito_op_1,        tito_op_1, tito_op_1]
    tito_op_list_2 = [tito_op_explicit, tito_op_1, tito_op_explicit]

    Ladder, outputs, inputs = build_braid(tito_op_list_1 = tito_op_list_1, tito_op_list_2 = tito_op_list_2, out_stride = 2)

    return Ladder(), outputs, inputs

# EVERYTHING ABOVE NEEDS EDITING
from python_csdl_backend.tests.create_single_test import run_test

def test_braid_simple():
    import numpy as np

    vals_dict = {
            'x_1_o_a': np.array([0.021, 0.021]),
            'x_1_o_b': np.array([0.0541, 0.0541]),
            'x_1_p_a': np.array([9.86, 9.86]),
            'x_1_p_b': np.array([1.55, 1.55]),
            'x_3_o_a': np.array([1.73630163e-08, 1.73630163e-08]),
            'x_3_o_b': np.array([2.07060146e-05, 2.07060146e-05]),
            'x_3_p_a': np.array([0.00042883, 0.00042883]),
            'x_3_p_b': np.array([3.79675746e-05, 3.79675746e-05]),
            'final_sum': np.array([0.00048752, 0.00048752]),
    }
    
    totals_dict = {
            ('final_sum', 'x_0_o_a'): np.array([[0.00095433, 0.        ],
    [0.        , 0.00095433]]),
            ('final_sum', 'x_0_o_b'): np.array([[0.00030697, 0.        ],
    [0.        , 0.00030697]]),
            ('final_sum', 'x_0_p_a'): np.array([[0.00046788, 0.        ],
    [0.        , 0.00046788]]),
            ('final_sum', 'x_0_p_b'): np.array([[0.00113985, 0.        ],
    [0.        , 0.00113985]]),
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
            ('x_1_p_a', 'x_0_o_b'): np.array([[1., 0.],
    [0., 1.]]),
            ('x_1_p_a', 'x_0_p_a'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_1_p_a', 'x_0_p_b'): np.array([[6.2, 0. ],
    [0. , 6.2]]),
            ('x_1_p_b', 'x_0_o_a'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_1_p_b', 'x_0_o_b'): np.array([[3.1, 0. ],
    [0. , 3.1]]),
            ('x_1_p_b', 'x_0_p_a'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_1_p_b', 'x_0_p_b'): np.array([[0.5, 0. ],
    [0. , 0.5]]),
            ('x_3_o_a', 'x_0_o_a'): np.array([[3.47260326e-08, 0.00000000e+00],
    [0.00000000e+00, 3.47260326e-08]]),
            ('x_3_o_a', 'x_0_o_b'): np.array([[3.64869876e-08, 0.00000000e+00],
    [0.00000000e+00, 3.64869876e-08]]),
            ('x_3_o_a', 'x_0_p_a'): np.array([[2.1747709e-08, 0.0000000e+00],
    [0.0000000e+00, 2.1747709e-08]]),
            ('x_3_o_a', 'x_0_p_b'): np.array([[1.6518894e-08, 0.0000000e+00],
    [0.0000000e+00, 1.6518894e-08]]),
            ('x_3_o_b', 'x_0_o_a'): np.array([[2.07060437e-05, 0.00000000e+00],
    [0.00000000e+00, 2.07060437e-05]]),
            ('x_3_o_b', 'x_0_o_b'): np.array([[2.10005972e-06, 0.00000000e+00],
    [0.00000000e+00, 2.10005972e-06]]),
            ('x_3_o_b', 'x_0_p_a'): np.array([[9.86002954e-06, 0.00000000e+00],
    [0.00000000e+00, 9.86002954e-06]]),
            ('x_3_o_b', 'x_0_p_b'): np.array([[1.30200185e-05, 0.00000000e+00],
    [0.00000000e+00, 1.30200185e-05]]),
            ('x_3_p_a', 'x_0_o_a'): np.array([[0.00085765, 0.        ],
    [0.        , 0.00085765]]),
            ('x_3_p_a', 'x_0_o_b'): np.array([[0.00018997, 0.        ],
    [0.        , 0.00018997]]),
            ('x_3_p_a', 'x_0_p_a'): np.array([[0.00041044, 0.        ],
    [0.        , 0.00041044]]),
            ('x_3_p_a', 'x_0_p_b'): np.array([[0.00106226, 0.        ],
    [0.        , 0.00106226]]),
            ('x_3_p_b', 'x_0_o_a'): np.array([[7.59351492e-05, 0.00000000e+00],
    [0.00000000e+00, 7.59351492e-05]]),
            ('x_3_p_b', 'x_0_o_b'): np.array([[0.00011486, 0.        ],
    [0.        , 0.00011486]]),
            ('x_3_p_b', 'x_0_p_a'): np.array([[4.75555485e-05, 0.00000000e+00],
    [0.00000000e+00, 4.75555485e-05]]),
            ('x_3_p_b', 'x_0_p_b'): np.array([[6.45581599e-05, 0.00000000e+00],
    [0.00000000e+00, 6.45581599e-05]]),
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

    m, outputs, inputs = get_model()
    import python_csdl_backend
    import numpy as np

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    sim = python_csdl_backend.Simulator(m, comm = comm, display_scripts=1, checkpoints = 0, save_vars=outputs)
    sim.run()


    of_vectors = {}
    # outputs = ['x_1_o_a']
    for of in outputs:
        of_vectors[of] = np.ones(sim[of].shape)
    x = sim.compute_vector_jacobian_product(of_vectors, ['x_0_p_a'])
    # x = sim.compute_totals(outputs, ['x_0_p_a'])
    print(x)
    # test_braid_mixed()