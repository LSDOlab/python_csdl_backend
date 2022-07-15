from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class QuatrotvecSample(csdl.Model):

    def define(self):

        shape = (2,)
        quatshape = shape + (4,)
        vecshape = shape + (3,)
        quatval = np.reshape(np.arange(np.prod(quatshape)), (quatshape))
        vecval = np.reshape(np.arange(np.prod(vecshape)), vecshape)

        quat = self.create_input('x', val=quatval)
        vec = self.create_input('w', val=vecval)

        y = csdl.quatrotvec(quat, vec)

        self.register_output('y', y)


def test_quatrotvec():
    # nn = 1
    # in_val = 0.5*np.ones(nn)
    # val = np.quatrotvec(in_val)
    # vals_dict = {'y': val}

    # total = np.diag((1.0 - np.quatrotvec(in_val)**2).flatten())
    # totals_dict = {('y', 'x'): total}

    totals_dict = {}
    vals_dict = {}
    run_test(
        QuatrotvecSample(),
        outs=['y'],
        ins=['x'],
        name='test_quatrotvec',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


# def test_quatrotvec_large():
#     nn = (10, 10)
#     in_val = 0.5*np.ones(nn)
#     val = np.quatrotvec(in_val)
#     vals_dict = {'y': val}

#     total = np.diag((1.0 - np.quatrotvec(in_val)**2).flatten())
#     totals_dict = {('y', 'x'): total}

#     run_test(
#         QuatrotvecSample(scalability_param=nn),
#         outs=['y'],
#         ins=['x'],
#         name='test_quatrotvec_large',
#         vals_dict=vals_dict,
#         totals_dict=totals_dict)
