from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class ScalarExtremumMaxSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        m = 2
        n = 3
        o = 4
        p = 5
        q = 6

        # Shape of a tensor
        tensor_shape = (m, n, o, p, q)

        num_of_elements = np.prod(tensor_shape)
        # Creating the values of the tensor
        val = np.arange(num_of_elements).reshape(tensor_shape)

        # Declaring the tensor as an input
        ten = self.declare_variable('x', val=val)

        # Computing the maximum across the entire tensor, returns single value
        ma = self.register_output('y', csdl.max(ten))

        p = self.register_output('y_out', csdl.expand(csdl.max(ten), shape=(3, 3, 3, 3, 3, 3)))


def test_scalarextremummax():

    m = 2
    n = 3
    o = 4
    p = 5
    q = 6

    tensor_shape = (m, n, o, p, q)
    num_of_elements = np.prod(tensor_shape)
    tensor = np.arange(num_of_elements).reshape(tensor_shape)

    # SCALAR MIN
    val = np.max(tensor)
    vals_dict = {'y': val}

    # total = np.eye(nn)
    totals_dict = {}

    run_test(
        ScalarExtremumMaxSample(scalability_param=1),
        outs=['y'],
        ins=['x'],
        name='test_scalarextremummax',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


if __name__ == '__main__':
    import python_csdl_backend

    class ScalarExtremumMaxSample2(csdl.Model):

        def initialize(self):
            self.parameters.declare('scalability_param')

        def define(self):
            m = 2
            n = 3
            o = 4
            p = 5
            q = 6

            # Shape of a tensor
            tensor_shape = (m, n, o, p, q)

            num_of_elements = np.prod(tensor_shape)
            # Creating the values of the tensor
            val = np.arange(num_of_elements).reshape(tensor_shape)
            val = val*0.0
            val[0] = 3.0
            # val[0] = 3.0

            # Declaring the tensor as an input
            ten = self.declare_variable('x', val=val)
            ten2 = self.declare_variable('x2', val=val.flatten())

            # Computing the maximum across the entire tensor, returns single value
            ma = self.register_output('y', csdl.max(ten))

            p = self.register_output('y_out', csdl.expand(csdl.max(ten), shape=(3, 3, 3, 3, 3, 3)))
            p2 = self.register_output('y2_out', csdl.max(ten2))

    sim = python_csdl_backend.Simulator(ScalarExtremumMaxSample2(scalability_param=1))
    sim.run()
    # sim.compute_totals(of = 'y_out', wrt = 'x')
    sim.check_totals(of=['y_out', 'y2_out'], wrt='x', compact_print=False)
