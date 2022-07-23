import numpy as np
SPARSITY_CUTOFF = 0.5

def sparse_matrix(data, rows, cols, shape=None, sptype='csc'):

    # creates string of scipy sparse matrix
    if shape:
        string = f'sp.{sptype}_matrix(({data}, ({rows}, {cols})), shape = {shape})'
    else:
        string = f'sp.{sptype}_matrix(({data}, ({rows}, {cols})))'

    return string

def get_sparsity(num_nnz_elements, num_outs, num_ins):
    """
    returns ratio of nnz to nz
    """
    return num_nnz_elements/(num_outs*num_ins)
