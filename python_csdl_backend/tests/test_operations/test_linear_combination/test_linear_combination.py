from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class LinearCombinationSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        w = self.create_input('w', val=3*np.ones(scalability_param))
        x = self.create_input('x', val=2*np.ones(scalability_param))

        y = 2*w+4*x

        self.register_output('y', y)
        self.register_output('y2', y-y)


def test_linear_combination():
    nn = 1
    vals_dict = {'y': 6*np.ones(nn) + 8*np.ones(nn), 'y2': np.zeros(nn)}
    totals_dict = {
        ('y', 'x'): 4*np.eye(nn),
        ('y', 'w'): 2*np.eye(nn),
    }

    run_test(
        LinearCombinationSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_linear_combination',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_linear_combination_large():
    nn = (10, 10)
    vals_dict = {'y': 6*np.ones(nn) + 8*np.ones(nn), 'y2': np.zeros(nn)}
    totals_dict = {
        ('y', 'x'): 4*np.eye(100),
        ('y', 'w'): 2*np.eye(100),
    }

    run_test(
        LinearCombinationSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_linear_combination_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


if __name__ == '__main__':
    # test_linear_combination()
    # test_linear_combination_large()

    import python_csdl_backend
    import csdl
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    sim_lite = python_csdl_backend.Simulator(LinearCombinationSample(scalability_param=(10, 10)), comm = comm, display_scripts=1)
    sim_lite.run()


    of_vectors = {}
    outs_check_vjp =['y']
    for i, key in enumerate(outs_check_vjp):
        of_vectors[key] = np.zeros(sim_lite[key].shape).flatten()
        if i == 0:
            check_key = key
            of_vectors[key][0] = 1.0

    in_vars = [in_name for in_name in sim_lite.variable_info['leaf_start'].keys()]
    # in_vars = ['x']

    vjp_dict = sim_lite.compute_vector_jacobian_product(of_vectors=of_vectors, wrt=in_vars)

    # Lets make sure that the first row of the derivatives is equal
    check_dict = sim_lite.compute_totals(of=outs_check_vjp, wrt=in_vars)
    for key_deriv in check_dict:
        if key_deriv[0] != check_key:
            continue

        if isinstance(check_dict[key_deriv], np.ndarray):
            check_vector = check_dict[key_deriv][0,:]
        else:
            check_vector = check_dict[key_deriv].toarray()[0,:]
        
        if comm.rank == 1:
            print(key_deriv, check_vector.shape)
            print(key_deriv, vjp_dict[key_deriv].shape)

        np.testing.assert_almost_equal(
            check_vector.flatten(),
            vjp_dict[key_deriv].flatten(),
            decimal=5)