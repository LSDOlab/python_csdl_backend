from csdl import GraphRepresentation
from csdl.utils.prepend_namespace import prepend_namespace
import networkx as nx
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from csdl.rep.variable_node import VariableNode
from csdl.rep.operation_node import OperationNode
from python_csdl_backend.core.simulator import Simulator
import string
import time


def propagate_uq(
    rep: GraphRepresentation,
    rvs: Dict[str, np.ndarray] = {},
    method='NIPC',
    output_name='a',
):

    # # gemerate the einsums
    # uq_expand = {}
    # num_nodes = rvs[list(rvs.keys())[0]].shape[0]

    # for i in range(len(rv_list)):
    #    uq_expand[rv_list[i]] = build_expansion_func(i, len(rv_list), num_nodes)
    if method == 'MC' or method == 'kriging' or method == 'designed_quadrature':
        n_qp = rvs[list(rvs.keys())[0]].shape[0]
        n_rv = len(list(rvs.keys()))
        sim = Simulator(rep)
        output = np.ones((n_qp, 1))
        tt = 0
        for i in range(n_qp):
            for j in range(n_rv):
                sim[list(rvs.keys())[j]] = rvs[list(rvs.keys())[j]][i]
            t1 = time.time()
            sim.run()
            t2 = time.time()
            output[i][0] = sim[output_name]
            tt = tt + t2-t1
        return output, tt

    if method == 'NIPC':
        if not isinstance(rvs, dict):
            raise TypeError('rvs must be a dictionary')
        # if len(rvs.keys()) > 2:
        #    raise NotImplementedError('uq currently only works with 1 or 2 variables.')

        n_1d_qp = []
        n_qp = 1
        n_rv = len(list(rvs.keys()))
        for key in rvs.keys():
            n_1d_qp.append(len(rvs[key]))
            n_qp = n_qp * len(rvs[key])
        rv_samples = np.zeros((n_rv, n_qp))
        result_dp_index = np.arange(n_rv)
        for i in range(n_rv):
            input_dp_index = [i]
            rv_samples[i, :] = rv_expansion(
                rvs[list(rvs.keys())[i]], input_dp_index, result_dp_index, n_1d_qp)
        sim = Simulator(rep)
        output = np.ones((n_qp, 1))
        tt = 0
        for i in range(n_qp):
            for j in range(n_rv):
                sim[list(rvs.keys())[j]] = rv_samples[j][i]
            t1 = time.time()
            sim.run()
            t2 = time.time()
            output[i][0] = sim[output_name]
            tt = tt + t2-t1
        return output, tt

    if method == 'NIPC_ATE':
        """
        Performs uncertainty propagation using the ATE method of 
        random variables specified in rvs.

        Parameters:
        -----------
            rep: GraphRepresentation
                - GraphRepresentation of system model

            rvs: Dict
                - Dictionary of names of random variables in rep
                along with their ditributions.

        Returns:
        --------
            outputs: Dict
                - Dictionary of outputs and their values.
        """

        # PROCESS:
        # Build:
        #   - Partition nodes according to dependency data
        #   - Build simulator instance for each partitioned section

        """
        General?
            lvl 0      lvl 1       lvl 2    ....     lvl n

            o000  ---> o100
                            ----> o110
                ---> o010              --- ... --> o111
                            ----> o011
                ---> o001
                            ----> o101
        """

        """
        2 rv:
            lvl 0      lvl 1       lvl 2

            o000  ---> o10
                            ----> o11
                ---> o01        
        """

        # Execute:
        #   - for each simulator
        #   -- preallocate outputs arrays for each input

        # Errors
        if not isinstance(rvs, dict):
            raise TypeError('rvs must be a dictionary')
        # if len(rvs.keys()) > 4:
        #     raise NotImplementedError(
        #         'uq currently only works with 1,2,3 or 4 variables.')

        # update names of rep:
        for node in rep.flat_graph.nodes:
            if isinstance(node, VariableNode):
                node.full_name = prepend_namespace(node.namespace, node.name)

        # rv string ==> rv node
        rvs_with_node = {}
        rv_list = []
        for j, rv_name in enumerate(rvs):
            # find variable node of strings in variable argument
            if rv_name in rep.promoted_to_node:
                rv = rep.promoted_to_node[rv_name]
            elif rv_name in rep.unpromoted_to_promoted:
                rv = rep.unpromoted_to_promoted[rep.promoted_to_node[rv_name]]
            else:
                raise KeyError(f'cannot find variable {rv_name}')
            rvs_with_node[rv] = rvs[rv_name]
            rv_list.append(rv)

        # dependency data
        dep_data = rep.dependency_data(
            rvs.keys(),
            return_format='dictionary')
        # :====================:THIS ENTIRE PART NEEDS TO BE AUTOMATED:====================:
        num_nodes = rvs[list(rvs.keys())[0]].shape[0]  # need to be modifed !!
        num_1d_qp = []
        for i in range(len(rvs.keys())):
            num_1d_qp.append(rvs[list(rvs.keys())[i]].shape[0])

        partition_graph = nx.DiGraph()
        initial_node = (None, )

        # create all node keys
        nodes = create_all_node_keys(rv_list)
        nodes.append(initial_node)

        for node in nodes:
            build_node(partition_graph, node, rv_list,
                       dep_data, rep, num_1d_qp)

        # build all edges:
        build_edges(partition_graph)

        print('NUMBER OF EDGES: ', len(list(partition_graph.edges)))
        print('NUMBER OF NODES: ', len(list(partition_graph.nodes)))
        # exit()

        # gemerate the einsums
        uq_expand = {}
        num_nodes = rvs[list(rvs.keys())[0]].shape[0]

        for edge in partition_graph.edges:
            #print('\nEDGE', edge[0],edge[1])

            # input indices
            input_num_nodes = []
            # print('INPUT NODE:')
            input_list = []
            for rv_var_node in edge[0]:
                if rv_var_node == None:
                    continue
                else:
                    input_list.append(rv_list.index(rv_var_node))
                    input_num_nodes.append(
                        num_1d_qp[rv_list.index(rv_var_node)])
                    # print('i', rv_list.index(rv_var_node))

            # input indices
            output_num_nodes = []
            # print('OUTPUT NODE:')
            output_list = []
            for rv_var_node in edge[1]:
                if rv_var_node == None:
                    continue
                else:
                    output_list.append(rv_list.index(rv_var_node))
                    output_num_nodes.append(
                        num_1d_qp[rv_list.index(rv_var_node)])
                    # print('i', rv_list.index(rv_var_node))
            # print(input_list, output_list)
            # print('input', input_num_nodes)
            # print('output', output_num_nodes)
            partition_graph.edges[edge]['expansion_function'] = build_auto(
                input_list, output_list, input_num_nodes, output_num_nodes)
            # print(edge[1])
            # print(partition_graph.nodes[edge[0]]['dep_index'])

        # for i in range(len(rv_list)):
        #     uq_expand[rv_list[i]] = build_expansion_func(i, len(rv_list), num_nodes)

        # uq_expand = {}
        # for i in range(len(rv_list)):
        #     uq_expand[rv_list[i]] = build_expansion_func(i, len(rv_list), num_nodes)
        # :====================:THIS ENTIRE PART NEEDS TO BE AUTOMATED:====================:

        # nx.draw(partition_graph, with_labels = True)
        # plt.show()

        # Check to make sure the paritioning happened correctly
        # checks to make sure:
        # -- for a partition node i: outputs of all predecessor partitions are superset of inputs of i
        # -- for a partition node i: outputs of all predecessor partitions are disjoint
        # -- intersection of all partions' operations is empty set
        # -- set of all partitions' operations == set of original representation's operations
        all_ops = set()

        check_partition(partition_graph, initial_node, [], all_ops)
        num_ops = len([x for x in dep_data.keys()
                       if isinstance(x, OperationNode)])
        if len(all_ops) != num_ops:
            raise ValueError(
                'operation count mismatch between all partitions and original representation')

        # execute
        output = {}
        # t_ate =  = time.time()
        time_out = 0
        for node in nx.topological_sort(partition_graph):
            #print('COMPUTING PARTITION:', partition_graph.nodes[node]['string'])
            x, time_run = run_partition(
                partition_graph, node, rvs_with_node, uq_expand)
            output.update(x)
            time_out = time_out + time_run
            print('time:', time_out)
        # t2 = time.time()
        return output[output_name], time_out


def run_partition(
    partition_graph: nx.DiGraph,
    node: Tuple,
    rvs,
    uq_expand,
):
    """
    Parameters:
    -----------
        partition_graph: DiGraph
            - Graph where nodes represent partitioned parts of the graph.
        node: Tuple
            - node in partition_graph representing the current simulator instance to run.
    """

    node_current = partition_graph.nodes[node]
    sim = node_current['simulator']
    num_nodes = node_current['size']
    time_loop_run = 0
    for i in range(num_nodes):
        #print('\titeration ', i)

        if None not in node:
            # set inputs computed from predecessor
            for input_name, input_val in node_current['inputs'].items():
                sim[input_name] = input_val[i, :]

        # set random variable distribution inputs
        for rv in rvs:
            if rv.full_name in node_current['inputs'] and (len(node) == 1):
                sim[rv.full_name] = rvs[rv][i]

        # run simulation for node i
        t1 = time.time()
        sim.run()
        t2 = time.time()
        time_loop_run += (t2 - t1)

        # set output for i
        for output_name in node_current['outputs']:
            node_current['outputs'][output_name][i,
                                                 :] = sim[output_name].copy()

    # all outputs should now be computed.
    # apply expansion function for each output and set as successor inputs
    for output_name in node_current['outputs']:
        # for successor in partition_graph.successors(node):
        # s_name = partition_graph.nodes[successor]['string']
        for edge in partition_graph.out_edges(node):
            successor = edge[1]
            rv_expansion = partition_graph.edges[edge]['expansion_function']

            if output_name in partition_graph.nodes[successor]['inputs']:
                #print(f'\toutput for {s_name}: ', output_name)

                new_input = rv_expansion(node_current['outputs'][output_name])
                # current_rv_set = set(node)
                # if None in node:
                #     current_rv_set = set()
                # successor_rv_set = set(successor)
                # for i, rv_node in enumerate(successor_rv_set.symmetric_difference(current_rv_set)):
                #     if i == 0:
                #         new_input = uq_expand[rv_node](node_current['outputs'][output_name])
                #     else:
                #         new_input = uq_expand[rv_node](new_input)

                # if partition_graph.nodes[successor]['inputs'][output_name].shape != new_input.shape:
                #     raise ValueError('shape mismatch')

                partition_graph.nodes[successor]['inputs'][output_name] = new_input

    return node_current['outputs'], time_loop_run


def check_partition(partition_graph, node, checked, all_ops):
    print('\nCHECKING PARTITION:', partition_graph.nodes[node]['string'])
    # all operations in current partition should be disjoint of all others:
    for op in partition_graph.nodes[node]['simulator'].system_graph.eval_graph:
        if isinstance(op, OperationNode):
            if op not in all_ops:
                all_ops.add(op)
            else:
                raise KeyError(f'duplicate operation in {node}: ', op)
    print('\tclear: no duplicate operations in current partition')

    # The inputs for this node should be union of outputs of predecessors
    # unless node == initial_node or one of the inputs are rv
    if None not in node:
        current_inputs = set(partition_graph.nodes[node]['inputs'].keys())
        all_pred_outputs = set()

        # if inputs contain a random variable, remove it from inputs to check
        for var in node:
            if var.full_name in current_inputs:
                current_inputs.remove(var.full_name)

        for pred in partition_graph.predecessors(node):
            pred_outputs = set(partition_graph.nodes[pred]['outputs'].keys())

            # print(partition_graph.nodes[pred]['string'], pred_outputs)
            if len(pred_outputs.intersection(all_pred_outputs)) != 0:
                print(pred)
                print(pred_outputs)
                print(all_pred_outputs)
                raise ValueError('overlapping outputs: ',
                                 pred_outputs.intersection(all_pred_outputs))

            all_pred_outputs.update(pred_outputs)
        print('\tclear: no overlapping outputs in current partition predecessors')

        if not all_pred_outputs.issuperset(current_inputs):
            diff = current_inputs - all_pred_outputs
            print(partition_graph.nodes[node]['string'], diff)
            print('current inputs', current_inputs)
            print('pred outputs', all_pred_outputs)
            raise KeyError(
                'input not subset of outputs of partitioned predecessors functions')
        print('\tclear: inputs of current partition is subset of predecessors\' outputs')

    checked.append(node)
    # recursively check successors
    for successor in partition_graph.successors(node):
        if successor not in checked:
            check_partition(partition_graph, successor, checked, all_ops)


def build_node(
    pg,
    tuple_rv,
    rv_list,
    dd,
    rep,
    num_1d_qp,
):
    """
    Parameters:
    -----------
        pg: DiGraph
            - graph to add node to
        tuple_rv: 
            - node hash containing tuple of rv dependencies
        rvs:
            - user given rvs
        dd:
            - dependency data
    """

    # Add node hash
    pg.add_node(tuple_rv)
    # num_nodes = rvs[list(rvs.keys())[0]].shape[0]
    if None in tuple_rv:
        size = 1
    else:
        size = 1
        for rv_var_node in tuple_rv:
            size *= num_1d_qp[rv_list.index(rv_var_node)]
            # print(size)
        # OLD:
        # size = int(num_nodes**len(tuple_rv))
    pg.nodes[tuple_rv]['size'] = size

    # string name of node
    if None in tuple_rv:
        pg.nodes[tuple_rv]['string'] = 'None'
    else:
        name = ''
        for i, node in enumerate(tuple_rv):
            if i == 0:
                name += node.full_name
            else:
                name += (', ' + node.full_name)
        pg.nodes[tuple_rv]['string'] = name

    # build simulator
    pg.nodes[tuple_rv]['outputs'] = {}
    rep2 = copy(rep)
    rep2.flat_graph = nx.DiGraph()
    for node in dd:
        add_node = does_match(dd, node, tuple_rv)
        if add_node:
            if isinstance(node, OperationNode):
                rep2.flat_graph.add_edges_from(rep.flat_graph.in_edges(node))
                rep2.flat_graph.add_edges_from(rep.flat_graph.out_edges(node))
            else:
                pg.nodes[tuple_rv]['outputs']
            if None in tuple_rv:
                if isinstance(node, VariableNode):
                    rep2.flat_graph.add_node(node)

    sim = Simulator(rep2)
    pg.nodes[tuple_rv]['simulator'] = sim

    # check all outputs and inputs
    pg.nodes[tuple_rv]['inputs'] = {}
    for node in sim.system_graph.eval_graph.nodes:
        if isinstance(node, VariableNode):
            shape = node.var.shape
            if (len(list(rep2.flat_graph.successors(node))) == 0) or (len(list(rep2.flat_graph.successors(node))) < len(list(rep.flat_graph.successors(node)))):
                # if (len(list(rep2.flat_graph.successors(node))) == 0):
                if does_match(dd, node, tuple_rv):
                    pg.nodes[tuple_rv]['outputs'][node.full_name] = np.zeros(
                        (size, *shape))
            # if None in tuple_rv:
            #     pg.nodes[tuple_rv]['outputs'][node.full_name] = np.zeros((size, *shape))

            if len(list(rep2.flat_graph.predecessors(node))) == 0:
                pg.nodes[tuple_rv]['inputs'][node.full_name] = np.zeros(
                    (size, *shape))

    pg.nodes[tuple_rv]['dep_index'] = 1  # !!!!!


def does_match(dd, node, tuple_rv):

    matches = True
    for rv in dd[node]:
        if (dd[node][rv] and (rv in tuple_rv)) or ((not dd[node][rv]) and (rv not in tuple_rv)):
            continue
        else:
            matches = False

    return matches


def build_expansion_func(i, num_rvs, num_nodes):

    def expansion_func(x):
        out_shape = list(x.shape)
        out_shape[0] = num_nodes*out_shape[0]
        out_shape = tuple(out_shape)
        if i == 0:
            return np.reshape(np.einsum('i...,p...->pi...', x, np.ones((num_nodes, 1))), out_shape)
        elif i == 1:
            return np.reshape(np.einsum('i...,p...->ip...', x, np.ones((num_nodes, 1))), out_shape)
        else:
            return ValueError('error')

    return expansion_func


def build_auto(input_dp_index, result_dp_index, input_n_1d_qp, output_n_1d_qp):

    def expansion_func(x):
        index_letter_list = list(string.ascii_lowercase)
        n_qp = 1
        for i in output_n_1d_qp:
            n_qp = n_qp*i
        einsum_input_string1 = ''
        einsum_input_string2 = ''
        einsum_output_string = ''
        expand_index = []
        for index in result_dp_index:
            if index in input_dp_index:
                einsum_input_string1 += index_letter_list[index]
            if index not in input_dp_index:
                einsum_input_string2 += index_letter_list[index]
                expand_index.append(index)
            einsum_output_string += index_letter_list[index]
        einsum_input_string1 += '...'
        einsum_input_string2 += '...'
        einsum_output_string += '...'
        matrix1_size = []
        for i in expand_index:
            index = result_dp_index.index(i)
            matrix1_size.append(output_n_1d_qp[index])
            #print('missing index', index)
        matrix1_size.append(1)
        out_shape = list(x.shape)
        out_shape[0] = n_qp
        out_shape = tuple(out_shape)
        # print('####################')
        #print('input index', input_dp_index)
        #print('output index', result_dp_index)
        # print(f'{einsum_input_string1},{einsum_input_string2}->{einsum_output_string}')
        #print('x shape:', x.shape)
        # print(tuple(matrix1_size))
        #print('output shape: ', out_shape)
        if len(input_dp_index) <= 1:
            return np.reshape(np.einsum(f'{einsum_input_string1},{einsum_input_string2}->{einsum_output_string}', x, np.ones(tuple(matrix1_size))), out_shape)
        else:
            in_shape = list(x.shape)
            #print('original shape: ', in_shape)
            in_shape.pop(0)
            insert_indx = 0
            for j in range(len(input_dp_index)):
                in_shape.insert(insert_indx, input_n_1d_qp[j])
                insert_indx += 1
            in_shape = tuple(in_shape)
            #print('new shape: ', in_shape)
            return np.reshape(np.einsum(f'{einsum_input_string1},{einsum_input_string2}->{einsum_output_string}', np.reshape(x, in_shape), np.ones(tuple(matrix1_size))), out_shape)
    return expansion_func


def rv_expansion(rv1, input_dp_index, result_dp_index, n_1d_qp):
    index_letter_list = list(string.ascii_lowercase)
    n_qp = 1
    for i in result_dp_index:
        n_qp = n_qp*n_1d_qp[i]
    einsum_input_string1 = ''
    einsum_input_string2 = ''
    einsum_output_string = ''
    expand_index = []
    for index in result_dp_index:
        if index in input_dp_index:
            einsum_input_string1 += index_letter_list[index]
        if index not in input_dp_index:
            einsum_input_string2 += index_letter_list[index]
            expand_index.append(index)
        einsum_output_string += index_letter_list[index]
    einsum_input_string1 += '...'
    einsum_input_string2 += '...'
    einsum_output_string += '...'
    matrix1_size = []
    for i in expand_index:
        matrix1_size.append(n_1d_qp[i])
    matrix1_size.append(1)
    x1 = np.einsum(f'{einsum_input_string1},{einsum_input_string2}->{einsum_output_string}',
                   rv1, np.ones(tuple(matrix1_size)))
    x2 = np.reshape(x1, (n_qp))
    return x2


def create_all_node_keys(rv_list):
    # Return a list of all possible combinations of random variables

    from itertools import chain, combinations
    num_rvs = len(rv_list)
    s = list(range(0, num_rvs))

    # get powerset
    n_indices = chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

    # use rv_list elements instead of integers
    nodes = []
    for index_tuple in n_indices:
        node_temp = []

        for index in index_tuple:
            node_temp.append(rv_list[index])

        nodes.append(tuple(node_temp))

    return nodes


def build_edges(g):

    # n^2 algorithm, for each node u, check all other nodes v and check if u is a subset of v. If so, add edge
    for u in g.nodes:

        # set of rvs defined by u
        u_set = set(u)

        for v in g.nodes:

            # do not build self edges
            if u == v:
                continue
            # add edge from None to all
            if u == (None,):
                g.add_edge(u, v)
                continue

            # set of rvs defined by v
            v_set = set(v)

            # if v depends on u, add edge
            if u_set.issubset(v):
                g.add_edge(u, v)
