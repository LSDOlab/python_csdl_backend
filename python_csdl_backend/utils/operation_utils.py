SPARSE_SIZE_CUTOFF = 100
import numpy as np

def to_unique_list(object):
    '''
    'listifies' object
    '''

    if isinstance(object, list):
        return sorted([*set(object)])
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

def nl_solver_completion_status(
        nlsolver_type, 
        iter_num, 
        tol,
        did_converge):
    """
    returns status string of non-linear solver completion

    Parameters:
    -----------
        nlsolver_type: string of type of solver
        iter_num: iteration number when exit
        tol: user specified tolerance
        did_converge: obvious
    """
    
    if did_converge:
        return f'nonlinear solver: {nlsolver_type} converged in {iter_num} iterations.'
    else:
        return f'nonlinear solver: {nlsolver_type} did not converge to tol {tol} in {iter_num} iterations.'
