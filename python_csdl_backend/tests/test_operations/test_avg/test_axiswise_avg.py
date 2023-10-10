from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class AverageSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        n = self.parameters['scalability_param']

        # Declare a vector of length 3 as input
        v1 = self.create_input('v1', val=np.arange(n))

        # Output the average of all the elements of the vector v1
        y = csdl.average(v1)
        self.register_output('y', y)
        # print('Y SHAPE:', y.shape)


def test_average():
    n = 3
    val = np.arange(n)
    y = np.average(val)
    # print(y)
    # exit()
    vals_dict = {'y': y}

    # total = np.eye(nn)
    totals_dict = {}

    run_test(
        AverageSample(scalability_param=n),
        outs=['y'],
        ins=['v1'],
        name='test_axisaverage',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


class AxisaverageSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')

    def define(self):
        n = self.parameters['scalability_param']

        # Declare a vector of length 3 as input
        v1 = self.create_input('v1', val=np.arange(n).reshape((int(n/2), 2)))

        # Output the average of all the elements of the vector v1
        y = csdl.average(v1, axes=0)
        self.register_output('y', y)
        # print('Y SHAPE:', y.shape)


def test_axisaverage():
    n = 30
    val = np.arange(n).reshape((int(n/2), 2))
    y = np.average(val, axis=0)
    # print(y)
    # exit()
    vals_dict = {'y': y}

    # total = np.eye(nn)
    totals_dict = {}

    run_test(
        AxisaverageSample(scalability_param=n),
        outs=['y'],
        ins=['v1'],
        name='test_axisaverage',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


if __name__ == '__main__':
    test_average()
    # test_axisaverage()