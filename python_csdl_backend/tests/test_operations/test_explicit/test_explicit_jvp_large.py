from python_csdl_backend.tests.create_single_test import run_test
from csdl import CustomExplicitOperation
import csdl
import numpy as np


class Example(CustomExplicitOperation):
    def define(self):
        self.add_input('b', shape=(3,))
        self.add_input('a', shape=(2,))

        self.add_output('x')
        self.add_output('y', shape=(3,))

        self.declare_derivatives('y', 'b')
        self.declare_derivatives('y', 'a')
        self.declare_derivatives('x', 'a')
        self.declare_derivatives('x', 'b')

    def compute(self, inputs, outputs):
        outputs['x'] = inputs['a'][0] + inputs['b'][1]
        outputs['y'] = inputs['b']*np.array([inputs['a'][0], inputs['a'][1], 1.0])

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        # REV:
        # [da db] = [dx dy][px/pa px/pb]
        #                  [py/pa py/pb]

        b = inputs['b'].flatten()
        a = inputs['a'].flatten()

        pxpa = np.array([1.0, 0.0]).reshape((1, 2))
        pxpb = np.array([0.0, 1.0, 0.0]).reshape((1, 3))
        pypa = np.array([[b[0], 0.0], [0.0, b[1]], [0.0, 0.0]]).reshape((3, 2))
        pypb = np.diag([a[0], a[1], 1.0]).reshape((3, 3))
        A = np.block([[pxpa, pxpb], [pypa,  pypb]])
        x = np.block([d_outputs['x'].flatten(), d_outputs['y'].flatten()]).reshape((4,))

        d_in = x.dot(A)

        d_inputs['a'] = d_in[0:2]
        d_inputs['b'] = d_in[2:]


class ExplicitRunJVPLarge(csdl.Model):

    def define(self):

        a = self.create_input('a', val=np.array([2.0, 2.5]))
        b = self.create_input('b', val=np.array([3.0, 3.5, 4.0]))
        x, y = csdl.custom(b, a, op=Example())

        self.register_output('x', x)
        self.register_output('y', y)

        self.register_output('f', x + y[0] + y[1])
        self.register_output('f2', y*2.0)


def test_explicitJVP_large():
    vals_dict = {}
    totals_dict = {}
    run_test(
        ExplicitRunJVPLarge(), 
        outs = ['x', 'y', 'f', 'f2'], 
        ins = ['a', 'b'], 
        name='test_explicitJVP_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict,
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
