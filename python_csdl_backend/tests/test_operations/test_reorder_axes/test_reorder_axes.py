from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class ReorderAxesSample(csdl.Model):

    def define(self):
        # Declare mat as an input matrix with shape = (4, 2)
        mat = self.create_input(
            'M1',
            val=np.arange(4 * 2).reshape((4, 2)),
        )

        # Compute the transpose of mat
        self.register_output('axes_reordered_matrix',
                             csdl.reorder_axes(mat, 'ij->ji'))


def test_reorderaxes():
    val = np.arange(4 * 2).reshape((4, 2))
    desired_output = np.transpose(val)
    vals_dict = {'axes_reordered_matrix': desired_output}

    # total = np.eye(nn)
    totals_dict = {}

    run_test(
        ReorderAxesSample(),
        outs=['y'],
        ins=['M1'],
        name='test_reorderaxes',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
