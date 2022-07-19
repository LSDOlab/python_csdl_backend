from python_csdl_backend.tests.create_single_test import run_test
from csdl import CustomExplicitOperation
import csdl
import numpy as np


class Example(CustomExplicitOperation):
    def define(self):
        self.add_input('b')
        self.add_input('a')

        self.add_output('x')
        self.add_output('y')

        self.declare_derivatives('y', 'b')
        self.declare_derivatives('y', 'a')
        self.declare_derivatives('x', 'a')
        self.declare_derivatives('x', 'b')

    def compute(self, inputs, outputs):
        outputs['x'] = inputs['a'] + inputs['b']
        outputs['y'] = inputs['b']*inputs['a']

    def compute_derivatives(self, inputs, derivatives):
        derivatives['y', 'b'] = inputs['a']
        derivatives['y', 'a'] = inputs['b']
        derivatives['x', 'b'] = np.ones((1, 1))
        derivatives['x', 'a'] = np.ones((1, 1))


class Example2(CustomExplicitOperation):
    def define(self):
        self.add_input('c', shape=(10,))
        self.add_input('d', shape=(10,))
        self.add_input('e', shape=(1,))

        self.add_output('w', shape=(10,))
        self.add_output('v', shape=(10,))

        row_col = np.arange(10)
        self.declare_derivatives('v', 'c', rows=row_col, cols=row_col)
        self.declare_derivatives('v', 'd', rows=row_col, cols=row_col)

        val = np.ones(10)
        self.declare_derivatives('w', 'c', rows=row_col, cols=row_col, val=val)
        self.declare_derivatives('w', 'd', rows=row_col, cols=row_col, val=val)

    def compute(self, inputs, outputs):
        outputs['w'] = inputs['c'] + inputs['d']
        outputs['v'] = inputs['c']*inputs['d']

    def compute_derivatives(self, inputs, derivatives):
        derivatives['v', 'c'] = inputs['d']
        derivatives['v', 'd'] = inputs['c']


class Example3(CustomExplicitOperation):
    def define(self):
        self.add_input('n', shape=(3,))
        self.add_input('m', shape=(2,))

        self.add_output('u')
        self.add_output('z', shape=(3,))

        self.declare_derivatives('z', 'n')
        self.declare_derivatives('z', 'm')
        self.declare_derivatives('u', 'm')
        self.declare_derivatives('u', 'n')

    def compute(self, inputs, outputs):
        outputs['u'] = inputs['m'][0] + inputs['n'][1]
        outputs['z'] = inputs['n']*np.array([inputs['m'][0], inputs['m'][1], 1.0])

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        # REV:
        # [da db] = [dx dy][px/pa px/pb]
        #                  [py/pa py/pb]

        b = inputs['n'].flatten()
        a = inputs['m'].flatten()

        pxpa = np.array([1.0, 0.0]).reshape((1, 2))
        pxpb = np.array([0.0, 1.0, 0.0]).reshape((1, 3))
        pypa = np.array([[b[0], 0.0], [0.0, b[1]], [0.0, 0.0]]).reshape((3, 2))
        pypb = np.diag([a[0], a[1], 1.0]).reshape((3, 3))
        A = np.block([[pxpa, pxpb], [pypa,  pypb]])
        x = np.block([d_outputs['u'].flatten(), d_outputs['z'].flatten()]).reshape((4,))

        d_in = x.dot(A)

        d_inputs['m'] = d_in[0:2]
        d_inputs['n'] = d_in[2:]


class Explicit2Run(csdl.Model):

    def define(self):

        a = self.create_input('a', val=2.0)
        b = self.create_input('b', val=3.0)
        x, y = csdl.custom(b, a, op=Example())

        self.register_output('x', x)
        self.register_output('y', y)

        self.register_output('f', x+y)

        c = self.create_input('c', val=4.0*np.ones(10))
        e = self.create_input('e', val=5.0)

        self.register_output('d', c + csdl.expand(y, (10,)))
        d = self.declare_variable('d', shape=(10,))
        w, v = csdl.custom(c, d, e, op=Example2())
        # self.register_output('w', w)
        # self.register_output('v', v)

        self.register_output('n', csdl.expand((c[0] + w[0] + v[0]+x), (3,)))
        n = self.declare_variable('n', shape=(3,))
        m = self.create_input('m', val=5.0*np.ones(2,))
        u, z = csdl.custom(n, m, op=Example3())

        self.register_output('f2', csdl.expand(u[0] + z[0], (10,)) + w + csdl.expand(x, (10,)) + csdl.expand(a, (10,)))
        self.register_output('f3', 3*u[0])


def test_explicit2():
    vals_dict = {}
    totals_dict = {}

    run_test(
        Explicit2Run(), 
        outs = ['x', 'f2', 'f', 'f3'], 
        ins = ['a', 'b', 'c', 'e'], 
        name='test_explicit2_many',
        vals_dict = {},
        totals_dict = {},
        )


if __name__ == '__main__':

    # import csdl_om
    # sim = csdl_om.Simulator(ExplicitRun())

    import python_csdl_backend
    sim = python_csdl_backend.Simulator(ExplicitRun())
    sim.eval_instructions.save()
    sim.run()

    print(sim['x'])
    print(sim['y'])
    print(sim['f'])
