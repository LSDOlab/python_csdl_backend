from python_csdl_backend.tests.create_single_test import run_test
from python_csdl_backend.tests.test_model_build.test_model_trees.build_model_hierarchies import build_model

import csdl
import numpy as np


class TreeLargeSample(csdl.Model):

    def define(self):
        self.add(build_model(
            5, # Hierarchy size 
            num_calcs=1, # Number of variables per model 
            num_named_variables=1, # Number of variables per model
        ), promotes=[])
        # self.add(build_model(num = 0, num_vars=20000), promotes=[])

def test_tree():
    vals_dict = {}
    totals_dict = {}

    run_test(
        TreeLargeSample(),
        outs=[],
        ins=[],
        name='test_arcsin',
        vals_dict=vals_dict,
        totals_dict=totals_dict
    )

class TreeLargePromote(csdl.Model):

    def define(self):
        self.add(build_model(
            5, # Hierarchy size 
            num_calcs=1, # Number of variables per model 
            num_named_variables=1, # Number of variables per model
            build_type='declared w/ promote',
        ), promotes=[])
        # self.add(build_model(num = 0, num_vars=20000), promotes=[])

def test_tree_promote():
    vals_dict = {}
    totals_dict = {}

    run_test(
        TreeLargePromote(),
        outs=[],
        ins=[],
        name='test_arcsin',
        vals_dict=vals_dict,
        totals_dict=totals_dict
    )

class TreeLargeConnections(csdl.Model):

    def define(self):
        import random
        random.seed(10)
        conn_dict1 = {}
        conn_dict1['sources'] = set()
        conn_dict1['targets'] = set()
        self.add(build_model(
            4, # Hierarchy size 
            num_calcs=1, # Number of variables per model 
            num_named_variables=1, # Number of variables per model
            build_type='connected',
            conn_dict=conn_dict1,
            namespace = 'level_0.',
        ), name = 'level_0', promotes=[])

        connections = [
            ['level_0.level_4_0.level_3_1.level_2_0.level_1_0.x_0','level_0.level_4_1.x_0_d'],
            ['level_0.level_4_1.level_3_1.level_2_0.x_0','level_0.level_4_1.level_3_1.x_0_d'],
            ['level_0.level_4_1.level_3_0.x_0','level_0.level_4_1.level_3_1.level_2_1.x_0_d'],
            ['level_0.level_4_1.level_3_1.level_2_0.x_0','level_0.level_4_1.level_3_0.level_2_1.level_1_0.x_0_d'],
            ['level_0.level_4_0.x_0','level_0.level_4_1.level_3_1.level_2_0.x_0_d'],
            ['level_0.level_4_0.level_3_1.level_2_1.x_0','level_0.x_0_d'],
            ['level_0.x_0','level_0.level_4_1.level_3_0.x_0_d'],
            ['level_0.x_0','level_0.level_4_1.level_3_1.level_2_0.level_1_1.x_0_d'],
            ['level_0.level_4_0.level_3_0.level_2_1.x_0','level_0.level_4_0.level_3_0.level_2_0.x_0_d'],
            ['level_0.level_4_0.x_0','level_0.level_4_0.level_3_0.level_2_0.level_1_0.x_0_d'],
        ]
        for i in connections:
            self.connect(*i)

def test_tree_connections():
    vals_dict = {}
    totals_dict = {}

    run_test(
        TreeLargeConnections(),
        outs=['level_0.level_4_0.level_3_0.level_2_0.level_1_0.x_0_d', 'level_0.level_4_1.x_0_d'],
        ins=['level_0.level_4_0.x_0', 'level_0.x_0'],
        name='test_arcsin',
        vals_dict=vals_dict,
        totals_dict=totals_dict
    )

if __name__ == '__main__':

    import csdl
    import time
    import cProfile
    import pstats
    profiler = cProfile.Profile()
    profiler.enable()

    s = time.time()
    # g = csdl.GraphRepresentation(TreeLargeSample())
    g = csdl.GraphRepresentation(TreeLargeConnections())
    # g = csdl.GraphRepresentation(TreeLargePromote())
    # exit()
    print('TIME', time.time() - s)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    # stats.print_stats()
    profiler.dump_stats('output')
    
    import python_csdl_backend
    s = python_csdl_backend.Simulator(g)

    s.run()
    s.check_partials(compact_print=1)