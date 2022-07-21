from csdl import Operation
import numpy as np
# from csdl.core.output import Output
# from csdl.core.input import Input


def to_list(object):
    '''
    'listifies' object.
    '''

    if isinstance(object, list):
        return object
    if isinstance(object, tuple):
        return object
    else:
        return [object]


def get_only(dict):
    '''
    returns only key of dict
    '''

    if len(dict) > 1:
        raise ValueError(f'{dict} has multiple keys. Cannot get only key.')
    else:
        return next(iter(dict))


def get_deriv_name(of, wrt, partials=True):
    '''
    get the variable name of a derirvative. if partials == false, assume totals
    '''
    if partials:
        return f'p{of}_p{wrt}'
    else:
        return f'd{of}_d{wrt}'


def get_path_name(var):
    '''
    get the path name for code generation
    '''
    return f'path_to_{var}'


def get_csdl_type_string(csdl_node):

    if isinstance(csdl_node, Output):
        return 'Output'
    elif isinstance(csdl_node, Input):
        return 'Input'
    elif isinstance(csdl_node, Operation):
        return 'Operation'
    else:
        return f'{type(csdl_node)}'


def increment_id(id):
    '''
    given an id of 'v00055', return 'v00056'
    '''
    new_id = f'{id[:2]}{int(id[2:])+1}'
    return new_id


def lineup_string(string, max=10):
    """
    given string 'string', add spaces to end up until 'max' number of characters
    """
    sep = ''
    pad = max - len(string)
    return f'{string}{sep:<{pad}}'


def set_opt_upper_lower(
    value,
    var_name,
    shape,
    bound_type
):
    """
    Processes constraint and design variable upper and lower bounds
    """

    if isinstance(value, np.ndarray):  # if numpy array
        value_size = np.prod(value.shape)
        if value_size == np.prod(shape):
            return_array = value.reshape(shape)
        elif value_size == 1:
            return_array = np.ones(shape)*value.reshape((1,))
        else:
            raise ValueError(f'Optimization variable {var_name} {bound_type} bound shape is incompatible')
    elif isinstance(value, list): # if list, convert to numpy
        value = np.array(value)
        value_size = np.prod(value.shape)
        if value_size == np.prod(shape):
            return_array = value.reshape(shape)
        elif value_size == 1:
            return_array = np.ones(shape)*value.reshape((1,))
        else:
            raise ValueError(f'Optimization variable {var_name} {bound_type} bound shape is incompatible')
    elif value is None: # if None, give max bounds
        if bound_type == 'lower':
            return_array = np.ones(shape)*(-1.0e30)
        elif bound_type == 'upper':
            return_array = np.ones(shape)*(1.0e30)
    else: # Hopefully when it reaches here, its a scalar...
        return_array = np.ones(shape)*value

    return return_array
