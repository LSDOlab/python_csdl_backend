from csdl import Operation
import numpy as np
# from csdl.core.output import Output
# from csdl.core.input import Input


def to_unique_list(object):
    '''
    'listifies' object.
    '''

    if isinstance(object, list):
        return sorted([*set(object)])
    if isinstance(object, tuple):
        return sorted([*set(object)])
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

def get_reverse_seed(out_id):
    '''
    get the seed name for reverse mode
    '''
    return f'r_seed_{out_id}'

def get_path_name(var, out_id = None):
    '''
    get the path name for code generation
    '''

    if out_id is None:
        return f'path_to_{var}'
    else:
        return f'path_from_{out_id}_to_{var}'

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
    bound_type,
    scaler,
):
    """
    Processes constraint and design variable upper and lower bounds.

    scaler is a numpy array in the shape of shape.
    """
    if isinstance(value, np.ndarray):  # if numpy array
        value_size = np.prod(value.shape)
        if value_size == np.prod(shape):
            return_array = value.reshape(shape)*scaler
        elif value_size == 1:
            return_array = (np.ones(shape)*value.reshape((1,)))*scaler
        else:
            raise ValueError(f'Optimization variable {var_name} {bound_type} bound shape is incompatible')
    elif isinstance(value, list):  # if list, convert to numpy
        value = np.array(value)
        value_size = np.prod(value.shape)
        if value_size == np.prod(shape):
            return_array = value.reshape(shape)*scaler
        elif value_size == 1:
            return_array = (np.ones(shape)*value.reshape((1,)))*scaler
        else:
            raise ValueError(f'Optimization variable {var_name} {bound_type} bound shape is incompatible')
    elif value is None:  # if None, give max bounds
        if bound_type == 'lower':
            return_array = np.ones(shape)*(-1.0e30)
        elif bound_type == 'upper':
            return_array = np.ones(shape)*(1.0e30)
    else:  # Hopefully when it reaches here, its a scalar...
        return_array = (np.ones(shape)*value)*scaler

    return return_array


def set_scaler_array(
    scaler,
    var_name,
    shape,
):
    if isinstance(scaler, np.ndarray):  # if numpy array
        scaler_size = np.prod(scaler.shape)
        if scaler_size == np.prod(shape):
            return_array = scaler.reshape(shape)
        elif scaler_size == 1:
            return_array = np.ones(shape)*scaler.reshape((1,))
        else:
            raise ValueError(f'Optimization variable {var_name} scalar shape is incompatible')
    elif scaler is None:  # no scaler
        return_array = np.ones(shape)
    else:  # scalar scaler
        return_array = np.ones(shape)*scaler

    return return_array

def analyze_dict_memory(var_dict, name, sim_name):
    import scipy.sparse as sp
    import sys
    total_size = 0
    total_size_dict = {
        'partial matrices': 0 ,
        '_coeff_temp': 0,
        'states': 0,
        'adjoints': 0,
        'other': 0,
    }
    print(f"=========================================={sim_name}:{name}==========================================")
    for key in var_dict:
        if var_dict[key] is not None:
            var = var_dict[key]
            bytesize = 0
            vartype = type(var)

            # Estimate memory cost of each variable
            if sp.issparse(var):
                vartype = 'sparse'
                bytesize = var.data.nbytes
                total_size += bytesize
            elif isinstance(var, np.ndarray):
                vartype = 'dense'
                bytesize = var.nbytes
                total_size += bytesize
            elif isinstance(var, tuple):
                vartype = 'tuple'
                bytesize = sys.getsizeof(var)
                total_size += bytesize

            if bytesize > 10e6:
                print('\t',key, '\t',vartype, '\t\t',f"{bytesize:,} (LARGE OBJECT)")        
            else:
                print('\t',key, '\t',vartype, '\t\t',f"{bytesize:,}")
            if key[0:2] == 'pv':
                total_size_dict['partial matrices'] += bytesize
            elif key[0] == 'v':
                total_size_dict['states'] += bytesize
            elif 'path_from' in key:
                total_size_dict['adjoints'] += bytesize
            elif '_coeff_temp' in key:
                total_size_dict['_coeff_temp'] += bytesize
            else:
                total_size_dict['other'] += bytesize
    print(name, 'total size', f"{total_size:,}")
    for var_type in total_size_dict:
        print('-', f'{var_type} size\t', f"{total_size_dict[var_type]:,}")

    print(f"=========================================={sim_name}:{name}==========================================")
    