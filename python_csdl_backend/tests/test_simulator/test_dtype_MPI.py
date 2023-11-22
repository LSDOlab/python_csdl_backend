from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class Example(csdl.CustomExplicitOperation):

    def initialize(self):
        self.parameters.declare('sname', types=str)

    def define(self):
        self.sname = self.parameters['sname']

        self.add_input('x'+self.sname, shape = (2,2))

        self.add_output(self.sname, shape = (2,2))

        self.declare_derivatives(self.sname, 'x'+self.sname)

    def compute(self, inputs, outputs):
        outputs[self.sname] = inputs['x'+self.sname].astype('float32')

    def compute_derivatives(self, inputs, derivatives):
        derivatives[self.sname, 'x'+self.sname] = np.eye(4)

class TestDtypeModel(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')
        self.parameters.declare('dtype')

    def define(self):
        scalability_param = self.parameters['scalability_param']
        dtype = self.parameters['dtype']

        x = self.create_input('x', val=0.5*np.ones(scalability_param, dtype=dtype))

        x1 = self.register_output('x1_temp', x*1.0)
        x2 = self.register_output('x2_temp', x*1.0)
        x3 = self.register_output('x3_temp', x*1.0)
        x4 = self.register_output('x4_temp', x*1.0)

        y1 = csdl.custom(x1, op=Example(sname='1_temp'))#*x_const*x_const*x_const*x_const*x_const*x_const*x_const*x_const*x_const*x_const
        y2 = csdl.custom(x2, op=Example(sname='2_temp'))#*x_const*x_const*x_const*x_const*x_const*x_const*x_const*x_const*x_const*x_const
        y3 = csdl.custom(x3, op=Example(sname='3_temp'))#*x_const*x_const*x_const*x_const*x_const*x_const*x_const*x_const*x_const*x_const
        y4 = csdl.custom(x4, op=Example(sname='4_temp'))#*x_const*x_const*x_const*x_const*x_const*x_const*x_const*x_const*x_const*x_const

        self.register_output('y1', y1*1.0)
        self.register_output('y2', y2*1.0)
        self.register_output('y3', y3*1.0)
        self.register_output('y4', y4*1.0)

def test_dtype():
    nn = (2,2)
    dtype = 'float64'
    result = 0.5*np.ones(nn)
    model = TestDtypeModel(scalability_param=nn, dtype=dtype)

    import python_csdl_backend
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    sim = python_csdl_backend.Simulator(model, comm = comm, display_scripts=0)
    sim.run()

    np.testing.assert_almost_equal(
        sim['y1'].flatten(),
        result.flatten(),
        decimal=5)
    np.testing.assert_almost_equal(
        sim['y2'].flatten(),
        result.flatten(),
        decimal=5)
    np.testing.assert_almost_equal(
        sim['y3'].flatten(),
        result.flatten(),
        decimal=5)
    np.testing.assert_almost_equal(
        sim['y4'].flatten(),
        result.flatten(),
        decimal=5)

if __name__ == '__main__':

    test_dtype()
