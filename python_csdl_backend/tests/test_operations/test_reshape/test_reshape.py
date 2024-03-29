from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np
import pytest

class FlattenSample(csdl.Model):

    def initialize(self):
        self.parameters.declare('scalability_param')
        self.parameters.declare('type', default='method')

    def define(self):
        scalability_param = self.parameters['scalability_param']
        type_flat = self.parameters['type']

        size = np.prod(scalability_param)

        x = self.create_input(
            'x',
            val=np.arange(size).reshape(scalability_param))
        
        if type_flat == 'method':
            self.register_output('y', x.flatten())
        elif type_flat == 'reshape_method':
            self.register_output('y', ((x.reshape(scalability_param) + x.reshape(*scalability_param))/2.0).flatten())
        elif type_flat == 'error1':
            self.register_output('y', x.reshape(scalability_param,1))
        else:
            self.register_output('y', csdl.flatten(x))


def test_flatten():
    nn = (1,)
    in_val = np.arange(1)
    val = in_val.flatten()
    vals_dict = {
        'y': val
    }

    total = np.diag([1.0])
    totals_dict = {('y', 'x'): total}

    run_test(
        FlattenSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_sech',
        vals_dict=vals_dict,
        totals_dict=totals_dict)

def test_flatten_large():
    nn = (10, 5, 3)
    in_val = np.arange(150).reshape(nn)
    val = in_val.flatten()
    vals_dict = {'y': val}

    total = np.diag(np.ones((150,)))
    totals_dict = {('y', 'x'): total}

    run_test(
        FlattenSample(scalability_param=nn),
        outs=['y'],
        ins=['x'],
        name='test_sech_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)

def test_flatten_func():
    nn = (1,)
    in_val = np.arange(1)
    val = in_val.flatten()
    vals_dict = {
        'y': val
    }

    total = np.diag([1.0])
    totals_dict = {('y', 'x'): total}

    run_test(
        FlattenSample(scalability_param=nn, type = 'func'),
        outs=['y'],
        ins=['x'],
        name='test_sech',
        vals_dict=vals_dict,
        totals_dict=totals_dict)

def test_flatten_large_func():
    nn = (10, 5, 3)
    in_val = np.arange(150).reshape(nn)
    val = in_val.flatten()
    vals_dict = {'y': val}

    total = np.diag(np.ones((150,)))
    totals_dict = {('y', 'x'): total}

    run_test(
        FlattenSample(scalability_param=nn, type = 'func'),
        outs=['y'],
        ins=['x'],
        name='test_sech_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
    

def test_reshape_large_func():
    nn = (10, 5, 3)
    in_val = np.arange(150).reshape(nn)
    val = in_val.flatten()
    vals_dict = {'y': val}

    total = np.diag(np.ones((150,)))
    totals_dict = {('y', 'x'): total}

    run_test(
        FlattenSample(scalability_param=nn, type = 'reshape_method'),
        outs=['y'],
        ins=['x'],
        name='test_sech_large',
        vals_dict=vals_dict,
        totals_dict=totals_dict)
    

def test_reshape_error_func():
    nn = (10, 5, 3)
    in_val = np.arange(150).reshape(nn)
    val = in_val.flatten()
    vals_dict = {'y': val}

    total = np.diag(np.ones((150,)))
    totals_dict = {('y', 'x'): total}

    import python_csdl_backend
    with pytest.raises(TypeError):
        m = FlattenSample(scalability_param=nn, type = 'error1')
        python_csdl_backend.Simulator(m)