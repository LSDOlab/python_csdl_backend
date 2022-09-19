from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class ExampleMultipleTensor(csdl.Model):
    def define(self):
        n = 3
        m = 6
        p = 7
        q = 10

        # Declare a tensor of shape 3x6x7x10 as input
        T1 = self.declare_variable(
            'x1',
            val=np.arange(n * m * p * q).reshape((n, m, p, q)),
        )

        # Declare another tensor of shape 3x6x7x10 as input
        T2 = self.declare_variable(
            'x2',
            val=np.arange(n * m * p * q, 2 * n * m * p * q).reshape(
                (n, m, p, q)),
        )
        # Output the elementwise average of tensors T1 and T2
        self.register_output('y',
                             csdl.average(T1, T2))


def test_mult_tensor_average():
    n = 3
    m = 6
    p = 7
    q = 10

    T1 = np.arange(n * m * p * q).reshape((n, m, p, q))
    T2 = np.arange(n * m * p * q, 2 * n * m * p * q).reshape((n, m, p, q))

    y = (T1 + T2) / 2.
    # print(y)
    # exit()
    vals_dict = {'y': y}

    # total = np.eye(nn)
    totals_dict = {}

    run_test(
        ExampleMultipleTensor(),
        outs=['y'],
        ins=['x1', 'x2'],
        name='test_mult_tensor_average',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


if __name__ == '__main__':
    from python_csdl_backend import Simulator
    sim = Simulator(ExampleMultipleTensor())
    sim.run()

    n = 3
    m = 6
    p = 7
    q = 10

    T1 = np.arange(n * m * p * q).reshape((n, m, p, q))
    T2 = np.arange(n * m * p * q, 2 * n * m * p * q).reshape((n, m, p, q))

    y = (T1 + T2) / 2.

    print(np.linalg.norm(y - sim['y']))

    t = sim.check_totals(of=['y'], wrt=['x1', 'x2'])
    # print(t['y', 'x1']['analytical_jac'].shape)
    # print(t['y', 'x1']['fd_jac'].shape)
    t = sim.check_totals(of=['y'], wrt=['x1', 'x2'])
