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

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        # FWD:
        # [dx] = [xa xb] [da]
        # [dy]   [ya yb] [db]

        # REV:
        # [da db] = [dx dy][px/pa px/pb]
        #                  [py/pa py/pb]

        b = inputs['b'][0]
        a = inputs['a'][0]

        A = np.array([[1.0, 1.0], [b,  a]])
        x = np.array([d_outputs['x'][0], d_outputs['y'][0]])

        d_in = x.dot(A)

        d_inputs['a'] += d_in[0]
        d_inputs['b'] += d_in[1]


class ExplicitRunJVP(csdl.Model):

    def define(self):

        a = self.create_input('a', val=2.0)
        b = self.create_input('b', val=3.0)
        x, y = csdl.custom(b, a, op=Example())

        # self.register_output('x', x)
        # self.register_output('y', y)
        self.register_output('f', x+1)


def test_explicit_simple():
    vals_dict = {}
    totals_dict = {}
    run_test(
        ExplicitRunJVP(), 
        outs = ['f'], 
        ins = ['a', 'b'], 
        name='test_explicitJVP_simple',
        vals_dict=vals_dict,
        totals_dict=totals_dict,
        )


if __name__ == '__main__':

    # import csdl_om
    # sim = csdl_om.Simulator(ExplicitRun())

    import python_csdl_backend
    sim = python_csdl_backend.Simulator(ExplicitRunJVP())
    # sim.eval_instructions.save()
    sim.run()

    # print(sim['x'])
    # print(sim['y'])
    print(sim['f'])

    sim.check_partials()