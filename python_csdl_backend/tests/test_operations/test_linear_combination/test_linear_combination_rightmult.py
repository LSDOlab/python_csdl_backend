from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class LinearCombinationRightOperation(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')
        self.parameters.declare('right_type')

    def define(self):
        scalability_param = self.parameters['scalability_param']
        right_operator_type = self.parameters['right_type']

        w = self.create_input('w', val=3*np.ones(scalability_param))
        x = self.create_input('x', val=2*np.ones(scalability_param))

        if right_operator_type == '__rmul__':
            self.register_output('y2', (2*np.ones(scalability_param))*w*(3*np.ones(scalability_param)))
            self.register_output('y', (2*np.ones(scalability_param))*w+(4*np.ones(scalability_param))*x)
        elif right_operator_type == '__radd__':
            self.register_output('y2', (2*np.ones(scalability_param))+w)
            self.register_output('y', ((2*np.ones(scalability_param))+w)+((4*np.ones(scalability_param))+x))
        elif right_operator_type == '__rsub__':
            self.register_output('y2', (2*np.ones(scalability_param))-w)
            self.register_output('y', ((2*np.ones(scalability_param))-w)+((4*np.ones(scalability_param))-x))
        elif right_operator_type == '__rtruediv__':
            self.register_output('y2', (2*np.ones(scalability_param))/w)
            self.register_output('y', ((2*np.ones(scalability_param))/w)+((4*np.ones(scalability_param))/x))
        elif right_operator_type == 'random':
            # randomly mix up the order of the operations to make sure that there are no issues when combining operations
            n_array_1 = 2*np.ones(scalability_param)
            n_array_2 = 3*np.ones(scalability_param)

            n_array_1[1,2] = 7
            n_array_1[9,1] = 79
            n_array_1[1,7] = 8
            n_array_1[5,6] = 1


            n_array_2[1,1] = 6
            n_array_2[2,9] = 45
            n_array_2[7,7] = 19
            n_array_2[5,6] = 41

            self.register_output('y0', n_array_1/n_array_2/(w/x))
            self.register_output('y1', n_array_1*w + n_array_2)
            self.register_output('y2', n_array_1*w/n_array_2)
            self.register_output('y3', w+n_array_2/x)
            self.register_output('y4', 3.0*w*n_array_1+n_array_2/x*w/4.0)
            self.register_output('y5', n_array_2/w/x/n_array_1)
            self.register_output('y6', n_array_1/n_array_2/(w/(4.0/x)))
            self.register_output('y7', n_array_1/n_array_2/(w/(n_array_2/x)))
            self.register_output('y8', n_array_1/(n_array_2/w)/x)
            self.register_output('y9', n_array_1 - n_array_2/(w/(n_array_2-x)))
            self.register_output('y10', n_array_1-(n_array_2-w)/x)
        else:
            raise ValueError(f'right operator type {right_operator_type} is unknown.')
def test_linear_combination():
    nn = 1
    size = 1
    vals_dict = {
        'y': 6*np.ones(nn) + 8*np.ones(nn),
        'y2': 18*np.ones(nn),}
    totals_dict = {
        ('y', 'x'): 4*np.eye(size),
        ('y', 'w'): 2*np.eye(size),
        ('y2', 'x'): np.zeros((size,size)),
        ('y2', 'w'): 6*np.eye(size),
    }

    run_test(
        LinearCombinationRightOperation(scalability_param=nn, right_type = '__rmul__'),
        outs=['y'],
        ins=['x'],
        name='test_linear_combination',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_linear_combination_large():
    nn = (10, 10)
    size = 100
    vals_dict = {
        'y': 6*np.ones(nn) + 8*np.ones(nn),
        'y2': 18*np.ones(nn),}
    totals_dict = {
        ('y', 'x'): 4*np.eye(size),
        ('y', 'w'): 2*np.eye(size),
        ('y2', 'x'): np.zeros((size,size)),
        ('y2', 'w'): 6*np.eye(size),
    }

    run_test(
        LinearCombinationRightOperation(scalability_param=nn, right_type = '__rmul__'),
        outs=['y'],
        ins=['x'],
        name='test_linear_combination_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)

def test_linear_combination_radd():
    nn = 1
    size = 1
    vals_dict = {
        'y': 11*np.ones(nn),
        'y2': 5*np.ones(nn),
    }
    totals_dict = {
        ('y', 'x'): np.eye(nn),
        ('y', 'w'): np.eye(nn),
        ('y2', 'x'): np.zeros((size,size)),
        ('y2', 'w'): np.eye(nn),
    }

    run_test(
        LinearCombinationRightOperation(scalability_param=nn, right_type = '__radd__'),
        outs=['y'],
        ins=['x'],
        name='test_linear_combination',
        vals_dict=vals_dict,
        totals_dict=totals_dict)

def test_linear_combination_radd_large():
    nn = (10,10)
    size = 100
    vals_dict = {
        'y': 11*np.ones(nn),
        'y2': 5*np.ones(nn),
    }
    totals_dict = {
        ('y', 'x'): np.eye(size),
        ('y', 'w'): np.eye(size),
        ('y2', 'x'): np.zeros((size,size)),
        ('y2', 'w'): np.eye(size),
    }

    run_test(
        LinearCombinationRightOperation(scalability_param=nn, right_type = '__radd__'),
        outs=['y'],
        ins=['x'],
        name='test_linear_combination',
        vals_dict=vals_dict,
        totals_dict=totals_dict)

def test_linear_combination_rsub():
    nn = 1
    size = 1
    vals_dict = {
        'y': 1*np.ones(nn),
        'y2': -1*np.ones(nn),
    }
    totals_dict = {
        ('y', 'x'): -np.eye(nn),
        ('y', 'w'): -np.eye(nn),
        ('y2', 'x'): np.zeros((size,size)),
        ('y2', 'w'): -np.eye(nn),
    }

    run_test(
        LinearCombinationRightOperation(scalability_param=nn, right_type = '__rsub__'),
        outs=['y'],
        ins=['x'],
        name='test_linear_combination',
        vals_dict=vals_dict,
        totals_dict=totals_dict)



def test_linear_combination_rsub_large():
    nn = (10,10)
    size = 100
    vals_dict = {
        'y': 1*np.ones(nn),
        'y2': -1*np.ones(nn),
    }
    totals_dict = {
        ('y', 'x'): -np.eye(size),
        ('y', 'w'): -np.eye(size),
        ('y2', 'x'): np.zeros((size,size)),
        ('y2', 'w'): -np.eye(size),
    }

    run_test(
        LinearCombinationRightOperation(scalability_param=nn, right_type = '__rsub__'),
        outs=['y'],
        ins=['x'],
        name='test_linear_combination',
        vals_dict=vals_dict,
        totals_dict=totals_dict)


def test_linear_combination_rtruediv():
    nn = 1
    size = 1
    vals_dict = {
        'y': (2/3+2)*np.ones(nn),
        'y2': 2/3*np.ones(nn),
    }
    totals_dict = {
        ('y', 'x'): -np.eye(nn),
        ('y', 'w'): -2/9*np.eye(nn),
        ('y2', 'x'): np.zeros((size,size)),
        ('y2', 'w'): -2/9*np.eye(nn),
    }

    run_test(
        LinearCombinationRightOperation(scalability_param=nn, right_type = '__rtruediv__'),
        outs=['y'],
        ins=['x'],
        name='test_linear_combination',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
    

def test_linear_combination_large_rtruediv():
    nn = (10,10)
    size = 100
    vals_dict = {
        'y': (2/3+2)*np.ones(nn),
        'y2': 2/3*np.ones(nn),
    }
    totals_dict = {
        ('y', 'x'): -np.eye(size),
        ('y', 'w'): -2/9*np.eye(size),
        ('y2', 'x'): np.zeros((size,size)),
        ('y2', 'w'): -2/9*np.eye(size),
    }

    run_test(
        LinearCombinationRightOperation(scalability_param=nn, right_type = '__rtruediv__'),
        outs=['y'],
        ins=['x'],
        name='test_linear_combination',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
    

def test_linear_combination_large_rand():
    nn = (10,10)
    size = 100


    n_array_1 = 2*np.ones(nn)
    n_array_2 = 3*np.ones(nn)

    n_array_1[1,2] = 7
    n_array_1[9,1] = 79
    n_array_1[1,7] = 8
    n_array_1[5,6] = 1


    n_array_2[1,1] = 6
    n_array_2[2,9] = 45
    n_array_2[7,7] = 19
    n_array_2[5,6] = 41
    w = 3*np.ones(nn)
    x = 2*np.ones(nn)

    vals_dict = {
            'y0': n_array_1/n_array_2/(w/x),
            'y1': n_array_1*w + n_array_2,
            'y2': n_array_1*w/n_array_2,
            'y3': w+n_array_2/x,
            'y4': 3.0*w*n_array_1+n_array_2/x*w/4.0,
            'y5': n_array_2/w/x/n_array_1,
            'y6': n_array_1/n_array_2/(w/(4.0/x)),
            'y7': n_array_1/n_array_2/(w/(n_array_2/x)),
            'y8': n_array_1/(n_array_2/w)/x,
            'y9': n_array_1 - n_array_2/(w/(n_array_2-x)),
            'y10': n_array_1-(n_array_2-w)/x,
    }
    totals_dict = {}

    run_test(
        LinearCombinationRightOperation(scalability_param=nn, right_type = 'random'),
        outs=['y'],
        ins=['x'],
        name='test_linear_combination',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
    
if __name__ == '__main__':
    test_linear_combination_large_rand()