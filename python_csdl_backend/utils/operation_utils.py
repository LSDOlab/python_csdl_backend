SPARSE_SIZE_CUTOFF = 100
import numpy as np

def to_list(object):
    '''
    'listifies' object
    '''

    if isinstance(object, list):
        return object
    else:
        return [object]


def get_scalars_list(scalars, names):
    '''
    from csdl_om
    '''
    if isinstance(scalars, (int, float)):
        scalars = [scalars] * len(names)
    elif isinstance(scalars, (list)):
        pass
    elif isinstance(scalars, np.ndarray):
        scalars = list(scalars)

    return scalars


def list_to_argument_str(argin_list):
    '''
    Parameters:
    -----------
        argin_list: list
            list of strings to turn into argument:

    i.e: ['a', 'b', 'c'] --> 'a,b,c'
    '''
    return_arg = ''
    for i, arg_in in enumerate(argin_list):
        if i == (len(argin_list) - 1):
            return_arg += arg_in
        else:
            return_arg += arg_in + ', '
    return return_arg
