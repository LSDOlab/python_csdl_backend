from csdl import Operation
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
