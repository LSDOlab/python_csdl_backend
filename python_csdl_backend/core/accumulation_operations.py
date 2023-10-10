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
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n_range = list(range(1, 200, 10))
    size = 50000
    
    times = []
    for n in n_range:

        path_to = np.random.rand(n, size)
        diag_vals = np.ones((size,))

        import time as time

        num = 10
        start = time.time()
        for i in range(num):
            diag_mult(path_to, diag_vals)
        end = time.time()
        
        print(f'{n}: {f_time}')
        f_time = (end-start)/num
        times.append(f_time)

    plt.plot(n_range, times)
    plt.show()