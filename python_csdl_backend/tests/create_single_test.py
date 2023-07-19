# from csdl_om import Simulator as OmSimulator


def run_test(model, outs, ins, name='', vals_dict=None, totals_dict=None):

    for sparsity_case in ['auto', 'dense', 'sparse']:
        # Test no parallel, no checkpointing
        run_test_single(model, outs, ins, name, sparsity_case, vals_dict, totals_dict)

        # Test no parallel, yes checkpointing
        run_test_single(model, outs, ins, name, sparsity_case, vals_dict, totals_dict, checkpoints = True)
        
        # Do not run parallel if mpi4py is not installed
        run_parallel = True
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        except:
            run_parallel = False

        if run_parallel:
            # Test yes parallel, no checkpointing
            run_test_single(model, outs, ins, name, sparsity_case, vals_dict, totals_dict, comm)
            
            # Test yes parallel, yes checkpointing
            run_test_single(model, outs, ins, name, sparsity_case, vals_dict, totals_dict, comm, checkpoints = True)


def run_test_single(model, outs, ins, name, sparsity_case, vals_dict, totals_dict, comm = None, checkpoints = False):
    from python_csdl_backend import Simulator as LiteSimulator
    import numpy as np

    from copy import deepcopy
    model1 = deepcopy(model)
    model2 = deepcopy(model)

    from copy import deepcopy
    print(f'RUNNING CASE: {name=} {sparsity_case=} {comm=} {checkpoints=}')
    if comm is None:
        sim_lite = LiteSimulator(model1,
                                sparsity=sparsity_case,
                                analytics=0,
                                display_scripts=False,
                                checkpoints=checkpoints,
                                save_vars='all',
            )
    else:
        assert comm.size > 0
        sim_lite = LiteSimulator(model2,
                                sparsity=sparsity_case,
                                analytics=0,
                                # display_scripts=False,
                                display_scripts=False,
                                comm = comm,
                                algorithm='Sync Points Coarse',
                                checkpoints=checkpoints,
                                save_vars='all',
            )
        
    from csdl.lang.node import Node
    Node._count = 0
    # sim_lite.eval_instructions.save()
    sim_lite.run()

    # UNCOMMENT FOR ASSERT STRINGS (EVAL):
    # --------------------------------------------
    # print('\nvals_dict = {')
    # for output in outs:
    #     # uncomment for assert full output
    #     formatted_numpy = np.array2string(sim_lite[output], separator=', ')
    #     print(f'\t\'{output}\': np.array({formatted_numpy}),')

    #     # uncomment for assert norm
    #     # formatted_numpy = np.linalg.norm(sim[output])
    #     # print(f'np.testing.assert_almost_equal(np.linalg.norm(outputs_dict[\'{output}\']), {formatted_numpy})')
    # print('}')
    # exit()
    # --------------------------------------------
    
    # UNCOMMENT FOR ASSERT STRINGS:
    # --------------------------------------------
    # import scipy.sparse as sp
    # totals_dict = sim_lite.compute_totals(of = outs, wrt = ins)
    # print('\ntotals_dict = {')
    # for check in totals_dict:
    #     out_str = 'totals_dict'
    #     temp = totals_dict[check]
    #     if sp.issparse(temp):
    #         temp = temp.toarray()
    #     formatted_numpy = np.array2string(temp, separator=', ')
    #     print(f'\t{check}: np.array({formatted_numpy}),')

    # print('}')
    # exit()
    # --------------------------------------------

    # for out in outs:
    #     print(out, sim_lite[out])
    error_dict = sim_lite.check_partials(compact_print=True)
    
    # check values if given
    for key in vals_dict:
        np.testing.assert_almost_equal(
            sim_lite[key],
            vals_dict[key],
            decimal=5)
    #     print(key, sim_lite[key], vals_dict[key])
    # exit()

    # check partials and analytical derivatives if given
    if len(error_dict) == 0:
        raise ValueError('No derivatives to check')

    for key in error_dict:
        np.testing.assert_almost_equal(
            error_dict[key]['relative_error_norm'],
            0.0,
            decimal=5)

        # print(key, error_dict[key]['analytical_norm'])
        if key in totals_dict:
            np.testing.assert_almost_equal(
                error_dict[key]['analytical_jac'],
                totals_dict[key],
                decimal=5)