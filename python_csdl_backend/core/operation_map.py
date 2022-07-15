from csdl.operations.linear_combination import linear_combination
from csdl.operations.power_combination import power_combination
from csdl.operations.exp import exp
from csdl.operations.sin import sin
from csdl.operations.cos import cos
from csdl.operations.arctan import arctan
from csdl.operations.arcsin import arcsin
from csdl.operations.arccos import arccos
from csdl.operations.expand import expand
from csdl.operations.pnorm import pnorm
from csdl.operations.decompose import decompose
from csdl.operations.reshape import reshape
from csdl.operations.indexed_passthrough import indexed_passthrough
from csdl.operations.cross import cross
from csdl.operations.sum import sum
from csdl.operations.einsum import einsum
from csdl.operations.passthrough import passthrough
from csdl.operations.dot import dot
from csdl.operations.log import log
from csdl.operations.log10 import log10
from csdl.operations.cosec import cosec
from csdl.operations.sec import sec
from csdl.operations.cotan import cotan
from csdl.operations.cosech import cosech
from csdl.operations.sech import sech
from csdl.operations.cotanh import cotanh
from csdl.operations.tan import tan
from csdl.operations.sinh import sinh
from csdl.operations.cosh import cosh
from csdl.operations.tanh import tanh
from csdl.operations.print_var import print_var
from csdl.operations.matmat import matmat
from csdl.operations.matvec import matvec
from csdl.operations.sparsematmat import sparsematmat
from csdl.operations.quatrotvec import quatrotvec
from csdl.operations.transpose import transpose
from csdl.operations.rotmat import rotmat
from csdl.operations.outer import outer
from csdl.operations.reorder_axes import reorder_axes
# from csdl.operations.inner import inner
# from csdl.operations.average import average
# from csdl.operations.min import min
# from csdl.operations.max import max


from csdl import CustomExplicitOperation


from python_csdl_backend.operations.linear_combination import get_linear_combination_lite
from python_csdl_backend.operations.power_combination import get_power_combination_lite
from python_csdl_backend.operations.pnorm import get_pnorm_lite
from python_csdl_backend.operations.exp import get_exp_lite
from python_csdl_backend.operations.sin import get_sin_lite
from python_csdl_backend.operations.cos import get_cos_lite
from python_csdl_backend.operations.arctan import get_arctan_lite
from python_csdl_backend.operations.arcsin import get_arcsin_lite
from python_csdl_backend.operations.arccos import get_arccos_lite
from python_csdl_backend.operations.expand import get_expand_lite
from python_csdl_backend.operations.decompose import get_decompose_lite
from python_csdl_backend.operations.reshape import get_reshape_lite
from python_csdl_backend.operations.indexed_passthrough import get_indexed_passthrough_lite
from python_csdl_backend.operations.cross import get_cross_lite
from python_csdl_backend.operations.sum import get_sum_lite
from python_csdl_backend.operations.einsum import get_einsum_lite
from python_csdl_backend.operations.passthrough import get_passthrough_lite
from python_csdl_backend.operations.dot import get_dot_lite
from python_csdl_backend.operations.log import get_log_lite
from python_csdl_backend.operations.log10 import get_log10_lite
from python_csdl_backend.operations.cosec import get_cosec_lite
from python_csdl_backend.operations.sec import get_sec_lite
from python_csdl_backend.operations.cotan import get_cotan_lite
from python_csdl_backend.operations.tan import get_tan_lite
from python_csdl_backend.operations.cosech import get_cosech_lite
from python_csdl_backend.operations.sech import get_sech_lite
from python_csdl_backend.operations.cotanh import get_cotanh_lite
from python_csdl_backend.operations.sinh import get_sinh_lite
from python_csdl_backend.operations.cosh import get_cosh_lite
from python_csdl_backend.operations.tanh import get_tanh_lite

from python_csdl_backend.operations.print_var import get_print_var_lite
from python_csdl_backend.operations.matmat import get_matmat_lite
from python_csdl_backend.operations.matvec import get_matvec_lite
from python_csdl_backend.operations.sparsematmat import get_sparsematmat_lite
from python_csdl_backend.operations.quatrotvec import get_quatrotvec_lite
from python_csdl_backend.operations.transpose import get_transpose_lite
from python_csdl_backend.operations.rotmat import get_rotmat_lite
from python_csdl_backend.operations.outer import get_outer_lite
from python_csdl_backend.operations.reorder_axes import get_reorder_axes_lite
# from python_csdl_backend.operations.inner import get_inner_lite
# from python_csdl_backend.operations.average import get_average_lite
# from python_csdl_backend.operations.min import get_min_lite
# from python_csdl_backend.operations.max import get_max_lite


from python_csdl_backend.operations.implicit.implicit_operation import get_implicit_lite, get_implicit_custom_lite
from python_csdl_backend.operations.custom_explicit.custom_explicit import get_custom_explicit_lite

# a mapping taking csdl operations and returning a function to create a backend operation object given a csdl node
csdl_to_back_map = {
    linear_combination: get_linear_combination_lite,
    power_combination:  get_power_combination_lite,
    exp:  get_exp_lite,
    sin:  get_sin_lite,
    cos:  get_cos_lite,
    pnorm: get_pnorm_lite,
    arctan:  get_arctan_lite,
    arcsin: get_arcsin_lite,
    arccos: get_arccos_lite,
    expand:  get_expand_lite,
    decompose: get_decompose_lite,
    reshape:  get_reshape_lite,
    indexed_passthrough:  get_indexed_passthrough_lite,
    cross:  get_cross_lite,
    sum:  get_sum_lite,
    einsum:  get_einsum_lite,
    passthrough: get_passthrough_lite,
    # CustomExplicitOperation: get_custom_explicit_lite,
    dot: get_dot_lite,
    log: get_log_lite,
    log10: get_log10_lite,
    cosec: get_cosec_lite,
    sec: get_sec_lite,
    cotan: get_cotan_lite,
    tan: get_tan_lite,
    cosech: get_cosech_lite,
    sech: get_sech_lite,
    cotanh: get_cotanh_lite,
    sinh: get_sinh_lite,
    cosh: get_cosh_lite,
    tanh: get_tanh_lite,
    print_var: get_print_var_lite,
    matmat: get_matmat_lite,
    matvec: get_matvec_lite,
    sparsematmat: get_sparsematmat_lite,
    quatrotvec: get_quatrotvec_lite,
    transpose: get_transpose_lite,
    rotmat: get_rotmat_lite,
    outer: get_outer_lite,
    reorder_axes: get_reorder_axes_lite,
    # inner : get_inner_lite,
    # average : get_average_lite,
    # min : get_min_lite,
    # max : get_max_lite,
}

# Function that returns function that returns class


def get_backend_op(csdl_node):
    optype = type(csdl_node)
    return csdl_to_back_map[optype](csdl_node)


def get_backend_custom_explicit_op(csdl_node):
    return get_custom_explicit_lite(csdl_node)


def get_backend_implicit_op(csdl_node):
    return get_implicit_lite(csdl_node)


def get_backend_custom_implicit_op(csdl_node):
    return get_implicit_custom_lite(csdl_node)
