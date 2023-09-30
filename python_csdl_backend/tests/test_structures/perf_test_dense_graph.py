from python_csdl_backend.tests.create_single_test import run_test


def get_model(type=0):
    from python_csdl_backend.tests.test_structures.utils import build_ladder
    import numpy as np
    import csdl
    from csdl import Model
    import numpy as np

    if type == 0:
        num_vars1 = 500
        num_vars2 = 500
    elif type == 1:
        num_vars1 = 1
        num_vars2 = 18000
    elif type == 2:
        num_vars1 = 18000
        num_vars2 = 1
    else:
        raise ValueError('type must be 1,2 or 3')

    class Dense(csdl.Model):

        def define(self):

            var_list = []
            for i in range(num_vars1):
                var_list.append(self.create_input(name=f'x_{i}', val=i))

            out_list = []

            for i in range(num_vars2):
                out_list.append(i*csdl.sum(*var_list))

            self.register_output('output', csdl.sum(*out_list))

    inputs = ['x_0']
    outputs = ['output']
    return Dense(), outputs, inputs


def get_model_index():
    from python_csdl_backend.tests.test_structures.utils import build_ladder
    import numpy as np
    import csdl
    from csdl import Model
    import numpy as np

    class Dense(csdl.Model):

        def define(self):

            num = 10000
            x = self.create_input(name='input', val=np.ones((num)))

            for i in range(num):
                out = self.register_output(f'out_{i}', x[i])

    inputs = ['input']
    outputs = ['out_0']
    return Dense(), outputs, inputs


def test_dense_graph():
    import numpy as np

    vals_dict = {}

    totals_dict = {}

    m, outputs, inputs = get_model()

    run_test(
        m,
        outs=outputs,
        ins=inputs,
        name='test_ladder_implicit',
        vals_dict=vals_dict,
        totals_dict=totals_dict,
        check_partials=False)


def test_dense_graph2():
    import numpy as np

    vals_dict = {}

    totals_dict = {}

    m, outputs, inputs = get_model(1)

    run_test(
        m,
        outs=outputs,
        ins=inputs,
        name='test_ladder_implicit',
        vals_dict=vals_dict,
        totals_dict=totals_dict,
        check_partials=False)


def test_dense_graph3():
    import numpy as np

    vals_dict = {}

    totals_dict = {}

    m, outputs, inputs = get_model(2)

    run_test(
        m,
        outs=outputs,
        ins=inputs,
        name='test_ladder_implicit',
        vals_dict=vals_dict,
        totals_dict=totals_dict,
        check_partials=False)


def test_dense_graph_index():
    import numpy as np
    # import sys
    # sys.setrecursionlimit(100000)
    vals_dict = {}

    totals_dict = {}

    m, outputs, inputs = get_model_index()

    run_test(
        m,
        outs=outputs,
        ins=inputs,
        name='test_ladder_implicit',
        vals_dict=vals_dict,
        totals_dict=totals_dict,
        check_partials=False)


if __name__ == '__main__':
    import python_csdl_backend
    import time
    import cProfile
    import csdl

    # test_dense_graph()
    # exit()

    # import sys
    # sys.setrecursionlimit(100000)

    m, outputs, inputs = get_model(0)
    # m, outputs, inputs = get_model(2)
    # m, outputs, inputs = get_model(2)
    # m, outputs, inputs = get_model_index()

    import cProfile
    import pstats
    profiler = cProfile.Profile()
    profiler.enable()

    s = time.time()
    g = csdl.GraphRepresentation(m)
    # sim = python_csdl_backend.Simulator(m)
    print('TIME', time.time() - s)
    # test_dense_graph()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    # stats.print_stats()
    profiler.dump_stats('output')
