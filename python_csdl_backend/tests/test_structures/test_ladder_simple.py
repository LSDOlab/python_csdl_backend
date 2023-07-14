def get_model():
    from python_csdl_backend.tests.test_structures.utils import build_ladder
    import numpy as np 
    import csdl
    from csdl import Model

    def tiso_op(x, y, model):
        return x*y

    Ladder, outputs, inputs = build_ladder(30, siso_op = csdl.sin, tiso_op = tiso_op, out_stride = 5)

    return Ladder(), outputs, inputs

# EVERYTHING ABOVE NEEDS EDITING
from python_csdl_backend.tests.create_single_test import run_test

def test_ladder_1():
    import numpy as np
    vals_dict = {
            'x_5_o': np.array([[0.60322485],
    [0.60322485]]),
            'x_5_p': np.array([[0.59371457],
    [0.59371457]]),
            'x_10_o': np.array([[0.02609327],
    [0.02609327]]),
            'x_10_p': np.array([[0.46604341],
    [0.46604341]]),
            'x_15_o': np.array([[0.00040811],
    [0.00040811]]),
            'x_15_p': np.array([[0.39726867],
    [0.39726867]]),
            'x_20_o': np.array([[3.14064331e-06],
    [3.14064331e-06]]),
            'x_20_p': np.array([[0.35241642],
    [0.35241642]]),
            'x_25_o': np.array([[1.39821668e-08],
    [1.39821668e-08]]),
            'x_25_p': np.array([[0.32013549],
    [0.32013549]]),
            'x_29_o': np.array([[1.32812028e-10],
    [1.32812028e-10]]),
    }


    totals_dict = {
            ('x_10_o', 'x_0_o'): np.array([[0.02609327, 0.        ],
    [0.        , 0.02609327]]),
            ('x_10_o', 'x_0_p'): np.array([[-0.05678983,  0.        ],
    [ 0.        , -0.05678983]]),
            ('x_10_p', 'x_0_o'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_10_p', 'x_0_p'): np.array([[-0.06927652, -0.        ],
    [-0.        , -0.06927652]]),
            ('x_15_o', 'x_0_o'): np.array([[0.00040811, 0.        ],
    [0.        , 0.00040811]]),
            ('x_15_o', 'x_0_p'): np.array([[-0.00115262,  0.        ],
    [ 0.        , -0.00115262]]),
            ('x_15_p', 'x_0_o'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_15_p', 'x_0_p'): np.array([[-0.0423566, -0.       ],
    [-0.       , -0.0423566]]),
            ('x_20_o', 'x_0_o'): np.array([[3.14064331e-06, 0.00000000e+00],
    [0.00000000e+00, 3.14064331e-06]]),
            ('x_20_o', 'x_0_p'): np.array([[-1.03831437e-05,  0.00000000e+00],
    [ 0.00000000e+00, -1.03831437e-05]]),
            ('x_20_p', 'x_0_o'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_20_p', 'x_0_p'): np.array([[-0.0293573, -0.       ],
    [-0.       , -0.0293573]]),
            ('x_25_o', 'x_0_o'): np.array([[1.39821668e-08, 0.00000000e+00],
    [0.00000000e+00, 1.39821668e-08]]),
            ('x_25_o', 'x_0_p'): np.array([[-5.16005401e-08,  0.00000000e+00],
    [ 0.00000000e+00, -5.16005401e-08]]),
            ('x_25_p', 'x_0_o'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_25_p', 'x_0_p'): np.array([[-0.02190591, -0.        ],
    [-0.        , -0.02190591]]),
            ('x_29_o', 'x_0_o'): np.array([[1.32812028e-10, 0.00000000e+00],
    [0.00000000e+00, 1.32812028e-10]]),
            ('x_29_o', 'x_0_p'): np.array([[-5.24694631e-10,  0.00000000e+00],
    [ 0.00000000e+00, -5.24694631e-10]]),
            ('x_5_o', 'x_0_o'): np.array([[0.60322485, 0.        ],
    [0.        , 0.60322485]]),
            ('x_5_o', 'x_0_p'): np.array([[-0.70857895,  0.        ],
    [ 0.        , -0.70857895]]),
            ('x_5_p', 'x_0_o'): np.array([[0., 0.],
    [0., 0.]]),
            ('x_5_p', 'x_0_p'): np.array([[-0.14776063, -0.        ],
    [-0.        , -0.14776063]]),
    }

    m, outputs, inputs = get_model()

    run_test(
        m,
        outs=outputs,
        ins=inputs,
        name='test_ladder_1',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
    

if __name__ == '__main__':
    test_ladder_1()