from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class MatmatSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        scalability_param = self.parameters['scalability_param']

        in_val1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        in_val2 = np.array([[2.0, 1.0], [2.0, 4.0]])

        x = self.create_input('x', val=in_val1)
        w = self.create_input('w', val=in_val2)

        y = csdl.matmat(x, w)

        self.register_output('y', y)


def test_matmat():
    nn = (2, 2)
    in_val1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    in_val2 = np.array([[2.0, 1.0], [2.0, 4.0]])
    val = in_val1 @ in_val2
    vals_dict = {'y': val}

    totalx = np.array(
        [[2., 2., 0., 0.],
         [1., 4., 0., 0.],
         [0., 0., 2., 2.],
         [0., 0., 1., 4.]])
    totalw = np.array(
        [[1., 0., 2., 0.],
         [0., 1., 0., 2.],
         [3., 0., 4., 0.],
         [0., 3., 0., 4.]])
    totals_dict = {
        ('y', 'x'): totalx,
        ('y', 'w'): totalw,
    }

    run_test(
        MatmatSample(scalability_param=nn),
        outs=['y'],
        ins=['x', 'w'],
        name='test_matmat',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


class MatmatSample2(csdl.Model):

    def initialize(self):
        self.parameters.declare('val_1')
        self.parameters.declare('val_2')

    def define(self):
        v1 = self.parameters['val_1']
        v2 = self.parameters['val_2']

        x = self.create_input('x', val=v1)
        w = self.create_input('w', val=v2)

        y = csdl.matmat(x, w)

        self.register_output('y', y)


def test_matmat2():
    in_val1 = np.random.rand(20, 20)
    in_val2 = np.random.rand(20, 20)
    val = in_val1 @ in_val2
    vals_dict = {'y': val}

    # totalx = np.array(
    #     [[2., 2., 0., 0.],
    #      [1., 4., 0., 0.],
    #      [0., 0., 2., 2.],
    #      [0., 0., 1., 4.]])
    # totalw = np.array(
    #     [[1., 0., 2., 0.],
    #      [0., 1., 0., 2.],
    #      [3., 0., 4., 0.],
    #      [0., 3., 0., 4.]])
    totals_dict = {}

    run_test(
        MatmatSample2(val_1=in_val1, val_2=in_val2),
        outs=['y'],
        ins=['x', 'w'],
        name='test_matmat2',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
