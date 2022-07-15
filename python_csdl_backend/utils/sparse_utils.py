import numpy as np


def sparse_matrix(data, rows, cols, shape=None, sptype='csc'):

    # creates string of scipy sparse matrix
    if shape:
        string = f'sp.{sptype}_matrix(({data}, ({rows}, {cols})), shape = {shape})'
    else:
        string = f'sp.{sptype}_matrix(({data}, ({rows}, {cols})))'

    return string
