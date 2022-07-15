from python_csdl_backend.tests.create_single_test import create_single_test


def create_operation_test(model_class, outs, ins, name='', vals_dict=None, totals_dict=None, param_cases=[]):

    scalability_param_list = [1, 10]
    sparsity_cases =['auto', 'dense', 'sparse']

    for scalability_param in scalability_param_list:
        for sparsity_case in sparsity_cases:
            create_single_test(model_class(scalability_param=scalability_param), outs, ins, sparsity = sparsity_case , name=name)
