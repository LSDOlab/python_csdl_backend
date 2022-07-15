import numpy as np
import scipy.sparse as sp


def check_not_implemented_args(op, given_metadata_dict, type_str):
    """
    raises an error when specific arguments are given.

    given_metadata_dict: dictionary of metadata for with variable name as keys
    type_str: str of 'input', 'output', 'derivative'
    """

    # Raise error for derivative/input/output parameters not implemented for this backend
    if type_str == 'input':
        not_implemented = {
            'src_indices': None,
            'flat_src_indices': None,
            'units': None,
            'desc': '',
            'tags': None,
            'shape_by_conn': False,
            'copy_shape': None
        }
    elif type_str == 'output':
        not_implemented = {
            'res_units': None,
            'lower': None,
            'units': None,
            'desc': '',
            'tags': None,
            'shape_by_conn': False,
            'copy_shape': None,
            'upper': None,
            'ref': 1.0,
            'ref0': 0.0,
            'res_ref': 1.0,
            'distributed': None,
        }
    elif type_str == 'derivative':
        not_implemented = {
            'method': 'exact',
            'step': None,
            'form': None,
            'step_calc': None,
        }

    for var in given_metadata_dict:
        temp = given_metadata_dict[var]

        for key_dont in not_implemented:
            if temp[key_dont] != not_implemented[key_dont]:
                raise NotImplementedError(f'argument \'{key_dont}\' for CustomExplicitOperation has not been implemented in this backend. {type_str} \'{var}\' in {type(op)} cannot be processed.')


def process_custom_derivatives_metadata(derivative_dict, out_dict, in_dict):
    """
    processes derivative metadata.
    given derivative metadata dict, processes:
    - standard dense numpy
    - sparse rows and columns given
    - sparse rows and columns and vals given
    - derivative not declared (zeros)
    """

    for derivative_tuple in derivative_dict:
        given_rows = derivative_dict[derivative_tuple]['rows']
        given_cols = derivative_dict[derivative_tuple]['cols']
        given_val = derivative_dict[derivative_tuple]['val']

        size_out = np.prod(out_dict[derivative_tuple[0]]['shape'])
        size_in = np.prod(in_dict[derivative_tuple[1]]['shape'])

        derivative_dict[derivative_tuple]['size_out'] = size_out
        derivative_dict[derivative_tuple]['size_in'] = size_in

        if given_rows is not None and given_cols is not None:
            if given_val is None:
                derivative_dict[derivative_tuple]['backend_type'] = 'row_col_given'
            elif given_val is not None:
                derivative_dict[derivative_tuple]['backend_type'] = 'row_col_val_given'
                derivative_dict[derivative_tuple]['given_val'] = sp.csc_matrix((given_val, (given_rows, given_cols)), shape=(size_out, size_in))
        else:
            derivative_dict[derivative_tuple]['backend_type'] = 'standard'

    for out_str in out_dict:
        for in_str in in_dict:
            derivative_tuple = (out_str, in_str)
            if derivative_tuple not in derivative_dict:
                size_out = np.prod(out_dict[out_str]['shape'])
                size_in = np.prod(in_dict[in_str]['shape'])

                derivative_dict[derivative_tuple] = {}
                derivative_dict[derivative_tuple]['size_out'] = size_out
                derivative_dict[derivative_tuple]['size_in'] = size_in
                derivative_dict[derivative_tuple]['backend_type'] = 'row_col_val_given'
                derivative_dict[derivative_tuple]['given_val'] = sp.csc_matrix((size_out, size_in))


def prepare_compute_derivatives(derivative_meta):

    derivatives = {}

    # Set derivatives
    for derivative_tuple in derivative_meta:

        # If rows and cols are given, give a flat vector with size len(rows) or size len(cols)
        if derivative_meta[derivative_tuple]['backend_type'] == 'row_col_given':
            len_val = len(derivative_meta[derivative_tuple]['rows'])
            derivatives[derivative_tuple] = np.zeros((len_val, ))
        else:

            # Otherwise, give zeros of 2D jac matrix
            size_out = derivative_meta[derivative_tuple]['size_out']
            size_in = derivative_meta[derivative_tuple]['size_in']

            derivatives[derivative_tuple] = np.zeros((size_out, size_in))

    return derivatives


def postprocess_compute_derivatives(totals, derivative_meta):

    # Post-process user given derivatives
    for derivative_tuple in derivative_meta:

        size_out = derivative_meta[derivative_tuple]['size_out']
        size_in = derivative_meta[derivative_tuple]['size_in']

        if derivative_meta[derivative_tuple]['backend_type'] == 'row_col_val_given':
            # If the value is given in define, use that.
            totals[derivative_tuple] = derivative_meta[derivative_tuple]['given_val']
        elif derivative_meta[derivative_tuple]['backend_type'] == 'row_col_given':

            # If the rows and cols are given, create sparse matrix of only vals.
            given_rows = derivative_meta[derivative_tuple]['rows']
            given_cols = derivative_meta[derivative_tuple]['cols']
            totals[derivative_tuple] = sp.csc_matrix((totals[derivative_tuple], (given_rows, given_cols)), shape=(size_out, size_in))
        else:
            # If standard derivative, just use user-given derivatie directly.
            totals[derivative_tuple] = totals[derivative_tuple].reshape((size_out, size_in))
