from python_csdl_backend.tests.create_single_test import run_test
from python_csdl_backend.tests.test_model_build.test_model_trees.build_model_hierarchies import build_model

import csdl
import numpy as np

class TreeLargeSample(csdl.Model):

    def define(self):
        self.add(build_model(
            10, # Hierarchy size 
            num_calcs=5, # Number of variables per model 
            num_named_variables=5, # Number of variables per model
        ), promotes=[])
        # self.add(build_model(num = 0, num_vars=20000), promotes=[])


def test_tree():
    nn = 1
    vals_dict = {}
    totals_dict = {}

    run_test(
        TreeLargeSample(),
        outs=['y'],
        ins=['x'],
        name='test_arcsin',
        vals_dict=vals_dict,
        totals_dict=totals_dict)

if __name__ == '__main__':

    import csdl
    import time
    import cProfile
    import pstats
    profiler = cProfile.Profile()
    profiler.enable()

    s = time.time()
    g = csdl.GraphRepresentation(TreeLargeSample())
    print('TIME', time.time() - s)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    # stats.print_stats()
    profiler.dump_stats('output')