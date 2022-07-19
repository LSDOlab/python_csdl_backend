# from csdl_om import Simulator as OmSimulator
from python_csdl_backend import Simulator as LiteSimulator
import pytest
import numpy as np
import time
import copy


def run_test(model, outs, ins, name='', vals_dict=None, totals_dict=None):

    for sparsity_case in ['auto', 'dense', 'sparse']:
        run_test_single(model, outs, ins, name, sparsity_case, vals_dict, totals_dict)


def run_test_single(model, outs, ins, name, sparsity_case, vals_dict, totals_dict):

    model_lite = model

    sim_lite = LiteSimulator(model_lite,
                             sparsity=sparsity_case,
                             analytics=0,
                             display_scripts=False)
    # sim_lite.eval_instructions.save()
    sim_lite.run()
    error_dict = sim_lite.check_partials(compact_print=True)

    # check values if given
    for key in vals_dict:
        np.testing.assert_almost_equal(
            sim_lite[key],
            vals_dict[key],
            decimal=5)

    # check partials and analytical derivatives if given
    if len(error_dict) == 0:
        raise ValueError('No derivatives to check')

    for key in error_dict:
        np.testing.assert_almost_equal(
            error_dict[key]['relative_error_norm'],
            0.0,
            decimal=5)

        print(key, error_dict[key]['analytical_norm'])
        if key in totals_dict:
            np.testing.assert_almost_equal(
                error_dict[key]['analytical_jac'],
                totals_dict[key],
                decimal=5)
