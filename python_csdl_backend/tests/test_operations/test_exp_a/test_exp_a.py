from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np
import pytest

class ExampleExponentialConstantScalarVariableScalar(csdl.Model):
    """
    :param var: y
    """
    def define(self):
        a = 5.0 
        x = self.declare_variable('x', val=3)
        y = csdl.exp_a(a, x)
        self.register_output('y', y)

def test_eecsvs():
    run_test(
        ExampleExponentialConstantScalarVariableScalar(),
        outs=['y'],
        ins=['x'],
        name='test_expa',
        vals_dict={'y': np.array(5.0**3)},
        totals_dict={})

class ExampleExponentialConstantScalarVariableArray(csdl.Model):
    """
    :param var: y
    """
    def define(self):
        a = 5.0
        x = self.declare_variable('x', val=np.array([1,2,3]))
        y = csdl.exp_a(a, x)
        self.register_output('y', y)

def test_eecsva():
    run_test(
        ExampleExponentialConstantScalarVariableArray(),
        outs=['y'],
        ins=['x'],
        name='test_expa',
        vals_dict={'y': np.array([5.0**1, 5.0**2, 5.0**3])},
        totals_dict={})


class ExampleExponentialConstantArrayVariableArray(csdl.Model):
    """
    :param var: y
    """
    def define(self):
        x = self.declare_variable('x', val=np.array([1,2,3]))
        a = np.array([1,2,3])
        y = csdl.exp_a(a, x)
        self.register_output('y', y)


def test_eecava():
    run_test(
        ExampleExponentialConstantArrayVariableArray(),
        outs=['y'],
        ins=['x'],
        name='test_expa',
        vals_dict={'y': np.array([1**1, 2**2, 3**3])},
        totals_dict={})


class ErrorExponentialConstantArrayVariableScalar(csdl.Model):
    def define(self):
        x = self.declare_variable('x', val=2.0)
        a = np.array([1,2,3])
        y = csdl.exp_a(a, x)
        self.register_output('y', y) 

def test_eecavs():
    with pytest.raises(ValueError):
        run_test(
            ErrorExponentialConstantArrayVariableScalar(),
            outs=['y'],
            ins=['x'],
            name='test_expa',
            vals_dict={'y': np.array([1**2, 2**2, 3**2])},
            totals_dict={})
        
class ExampleExponentialVariableArrayVariableArray(csdl.Model):
    """
    :param var: y
    """
    def define(self):
        x = self.declare_variable('x', val=np.array([1,2,3]))
        a = self.declare_variable('a', val=np.array([2,3,4]))
        y = csdl.exp_a(a, x)
        self.register_output('y', y)

def test_eevava():
    run_test(
        ExampleExponentialVariableArrayVariableArray(),
        outs=['y'],
        ins=['x'],
        name='test_expa',
        vals_dict={'y': np.array([2**1, 3**2, 4**3])},
        totals_dict={})
    
class ExampleExponentialVariableScalarVariableArray(csdl.Model):
    """
    :param var: y
    """
    def define(self):
        x = self.declare_variable('x', val=np.array([2,3,4]))
        a = self.declare_variable('a', val=np.array([4]))
        y = csdl.exp_a(a, x)
        self.register_output('y', y)

def test_eevavs():
    run_test(
        ExampleExponentialVariableScalarVariableArray(),
        outs=['y'],
        ins=['x'],
        name='test_expa',
        vals_dict={'y': np.array([4**2, 4**3, 4**4])},
        totals_dict={})