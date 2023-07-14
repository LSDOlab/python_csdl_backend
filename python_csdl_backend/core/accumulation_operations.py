import numpy as np
numpy_class = np.ndarray

# Performs multiplication of elementwise operations.
def diag_mult(path_to, diag_vals):
    # if len(diag_vals.shape) != 1:
    #     raise ValueError('wrong shape')

    # print(diag_vals.shape,path_to.shape)
    # print(type(diag_vals), type(path_to))
    if isinstance(path_to, numpy_class):
        # return path_to*diag_vals
        # print(f'{comm.rank} dense trigger')
        return np.multiply(path_to,diag_vals)
    else:
        # print(f'{comm.rank}sparse trigger')
        return path_to.multiply(diag_vals)