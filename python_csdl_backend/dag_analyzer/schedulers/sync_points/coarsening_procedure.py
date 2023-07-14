from python_csdl_backend.dag_analyzer.graph_generator.sdag_to_op_only_graph import sdag_to_op_only_graph
from python_csdl_backend.dag_analyzer.utils import draw
# utils

# other:
import pickle
import networkx as nx
import random

# Compute "top-level" value for each node.
def longest_path_length(graph):
    """
    Assigns an 'L' attribute to each node in a directed acyclic graph, representing the longest path to reach that node from any source.

    Parameters:
        graph (networkx.DiGraph): a directed acyclic graph

    Returns:
        None
    """
    # Initialize 'L' attribute of all nodes to 0
    nx.set_node_attributes(graph, -1, 'L')

    # For each node in the topological sort order, set its 'L' attribute to the maximum 'L' value of its predecessors plus one
    for node in nx.topological_sort(graph):
        max_length = -1
        for predecessor in graph.predecessors(node):
            max_length = max(max_length, graph.nodes[predecessor]['L'])
        graph.nodes[node]['L'] = max_length+1

def get_valid_neighbors(node, graph, weight_max_cutoff = None):

    """
    Given a node in graph, returns a set of valid neighboring nodes that node can be contracted to.
    """

    node_object = graph.nodes[node]

    # Choose neighbors based on paper
    if node_object['nbbadneighbors'] == 1:
        valid_neighbors = set()

        potential_leader = node_object['leaderbadneighbors']
        if ((abs(graph.nodes[potential_leader]['max_level'] - node_object['L'])) <= 1) and ((abs(graph.nodes[potential_leader]['min_level'] - node_object['L'])) <= 1):
            valid_neighbors.add(potential_leader)
    elif node_object['nbbadneighbors'] > 1:
        valid_neighbors = set()
    else:
        # There are no bad neighbors at this point
        valid_neighbors = set()

        # If a neighbor has a top-level difference greater than 1, it is an invalid neighbor
        for neighbor in nx.all_neighbors(graph, node):
            neighbor_object = graph.nodes[neighbor]

            # First part of theorem 4.2
            if ((abs(neighbor_object['max_level'] - node_object['L'])) <= 1) and ((abs(neighbor_object['min_level'] - node_object['L'])) <= 1):
                # If first part of theorem 4.2 is valid:
                valid_neighbors.add(neighbor)

    # Remove valid neighbors that make the time too high:
    remove_neighbors = set()
    for neighbor in valid_neighbors:
        neighbor_object = graph.nodes[neighbor]

        if weight_max_cutoff:
            if node_object['weight'] + neighbor_object['weight'] > weight_max_cutoff:
                remove_neighbors.add(neighbor)
    for remove_neighbor in remove_neighbors:
        valid_neighbors.remove(remove_neighbor)

    return valid_neighbors

def determine_best_neighbor(node, valid_neighbors, graph):

    """
    Given a node and list of neighbors it can merge with, choose the one to merge with. 
    Probably lowest time in the future.
    """
    return random.choice(tuple(valid_neighbors))

# Use clustering algorithm: 'Algorithm 1' in https://epubs.siam.org/doi/abs/10.1137/18M1176865?mobileUi=0 
def cluster_dag_to_dag(graph, weight_max_cutoff):
    # initialize auxiliary data:
    for node in graph.nodes:
        node_object = graph.nodes[node]
        node_object['mark'] = False
        node_object['leader'] = node
        node_object['weight'] = node_object['weight']
        node_object['nbbadneighbors'] = 0 # number of neighbors to node that will break the rules if node is placed in a different cluster.
        node_object['leaderbadneighbors'] = -1

        node_object['max_level'] = node_object['L']
        node_object['min_level'] = node_object['L']

    for node in nx.topological_sort(graph):
        node_object = graph.nodes[node]
        # print('PROCESSING NODE: ', node)
        # print('    top level:', node_object['L'])
        if node_object['mark']:
            continue
        
        valid_neighbors = get_valid_neighbors(node, graph, weight_max_cutoff=weight_max_cutoff)
        # print('    valid neighbors:', valid_neighbors)

        # If no valid neighbors, leave node as a singleton
        if len(valid_neighbors) == 0:
            continue
        # At this point, we want to contract the node to a cluster.
        best_neighbor = determine_best_neighbor(node, valid_neighbors, graph)
        best_neighbor_object = graph.nodes[best_neighbor]
        # print('    merge with:', best_neighbor)

        leader = best_neighbor_object['leader']
        leader_object = graph.nodes[leader]

        node_object['leader'] = leader
        leader_object['weight'] = leader_object['weight'] + node_object['weight']
        leader_object['max_level'] = node_object['max_level'] = max(leader_object['max_level'], node_object['max_level'])
        leader_object['min_level'] = node_object['min_level'] = min(leader_object['min_level'], node_object['min_level'])
        # I guess we have merged 'node' to the cluster with leader 'leader'
        # print(valid_neighbors, best_neighbor, node_object['leader'])

        # Update information to neighbors
        for neighbor in nx.all_neighbors(graph,node):
            neighbor_object = graph.nodes[neighbor]
            # if abs(neighbor_object['L'] - node_object['L']) > 1:
            if ((abs(neighbor_object['max_level'] - node_object['min_level'])) > 1) and (((abs(neighbor_object['min_level'] - node_object['max_level']))) > 1):
                continue

            if neighbor_object['nbbadneighbors'] == 0:
                neighbor_object['nbbadneighbors'] = 1
                neighbor_object['leaderbadneighbors'] = leader
            elif (neighbor_object['nbbadneighbors'] == 1 )and (neighbor_object['leaderbadneighbors'] != leader):
                neighbor_object['nbbadneighbors'] = 2

        # If best neighbor was forming a singleton cluster before u's assignment
        # (don't really understand)
        # print(list(graph.neighbors(best_neighbor)), best_neighbor_object['mark'])
        if (best_neighbor_object['mark'] == False):
            # let best neighbor's neighbors know that it is not a singleton anymore
            for best_neighbor_neighbor in nx.all_neighbors(graph,best_neighbor):
                best_neighbor_neighbor_object = graph.nodes[best_neighbor_neighbor]
                # if abs(best_neighbor_object['L'] - best_neighbor_neighbor_object['L']) > 1:
                if ((abs(best_neighbor_object['max_level'] - best_neighbor_neighbor_object['min_level'])) > 1) and (((abs(best_neighbor_object['min_level'] - best_neighbor_neighbor_object['max_level']))) > 1):
                    continue


                if best_neighbor_neighbor_object['nbbadneighbors'] == 0:
                    best_neighbor_neighbor_object['nbbadneighbors'] = 1
                    best_neighbor_neighbor_object['leaderbadneighbors'] = leader
                elif (best_neighbor_neighbor_object['nbbadneighbors'] == 1 )and (best_neighbor_neighbor_object['leaderbadneighbors'] != leader):
                    best_neighbor_neighbor_object['nbbadneighbors'] = 2

            best_neighbor_object['mark'] = True
        node_object['mark'] = True

def contract_to_coarsened_dag(graph):
    # This takes in a graph after the clustering algorithm has been ran and returns a merged graph
    contracted_graph = graph.copy()

    leader_to_node_dict = {}

    for node in graph:
        node_object = graph.nodes[node]

        node_leader = node_object['leader']

        if node_leader not in leader_to_node_dict:
            leader_to_node_dict[node_leader] = set()
        leader_to_node_dict[node_leader].add(node)

    # for key in leader_to_node_dict:
    #     print(key)
    #     for merge in leader_to_node_dict[key]:
    #         print('\t', merge)
    print('NODE REDUCTION:', f'({len(graph.nodes)} -> {len(leader_to_node_dict)})')
    # print('OLD GRAPH # NODES:', len(graph.nodes))

    for leader in leader_to_node_dict:
        for merge in leader_to_node_dict[leader]:
            
            if merge == leader:
                continue

            # Store information on original graph:
            # accumulate merged operations to clustered node
            contracted_graph.nodes[leader]['operation_cluster'].update(contracted_graph.nodes[merge]['operation_cluster'])

            if contracted_graph.has_edge(leader, merge):
                contracted_graph.nodes[leader]['clustered_variables'].update(contracted_graph.edges[(leader, merge)]['edge_variables'])
            elif contracted_graph.has_edge(merge, leader):
                contracted_graph.nodes[leader]['clustered_variables'].update(contracted_graph.edges[(merge, leader)]['edge_variables'])
            contracted_graph.nodes[leader]['clustered_variables'].update(contracted_graph.nodes[merge]['clustered_variables'])
            # accumulate edge variables from clustering.
            # Kind of complicated:
            # - - look for shared predecessors of merged operation and leader operations
            # - - merge the `edge_variables` attribute set
            # - - repeat for share successors
            
            leader_predecessors = set(contracted_graph.predecessors(leader))
            merge_predecessors = set(contracted_graph.predecessors(merge))
            shared_predecessors = merge_predecessors.intersection(leader_predecessors)
            for shared_predecessor in shared_predecessors:
                edge_variables_to_add = contracted_graph.edges[(shared_predecessor, merge)]['edge_variables']
                contracted_graph.edges[(shared_predecessor, leader)]['edge_variables'].update(edge_variables_to_add)

            leader_successors= set(contracted_graph.successors(leader))
            merge_successors = set(contracted_graph.successors(merge))
            shared_successors = merge_successors.intersection(leader_successors)
            for shared_successor in shared_successors:
                edge_variables_to_add = contracted_graph.edges[(merge, shared_successor)]['edge_variables']
                contracted_graph.edges[(leader, shared_successor)]['edge_variables'].update(edge_variables_to_add)


            # merge nodes
            nx.contracted_nodes(
                G = contracted_graph,
                u = leader, # u is kept 
                v = merge, # v is merged into u and deleted
                self_loops = False,
                copy = False,
            )


    # if not nx.is_directed_acyclic_graph(contracted_graph):
    #     print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ NOT DAG @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    #     raise ValueError('NOT DAG')
    return contracted_graph


def coarsen_graph(
        o_sdag,
        MAX_CLUSTER_WEIGHT_PERCENTAGE,
        total_weight,
        make_plots = False
    ):
    print('\n')

    # operation only graph
    weight_max_cutoff = MAX_CLUSTER_WEIGHT_PERCENTAGE*total_weight

    # try to apply algorithm from "Acyclic Partitioning of Large Directed Acyclic Graphs"
    # https://epubs.siam.org/doi/abs/10.1137/18M1176865?mobileUi=0 

    # coarsen graph procedure:
    # -- TODO: Heuristic for best neighbor
    # -- TODO: Heutistic for node processing order 
    # - Data structure: Digraph:
    # - Rules for correctness:
    # - - Every node v_c \in CDAG (coarsened/clustered dag) specifies a set of operations O_v_c
    # - - o is an element of V_o for all o \in O_v_c for all v_c \in V_c
    # - - Union({O_v_c}_{v_c \in V_c}) = V_o
    # - - Every edge e = (v, u) \in E_c contains a set of variables Y_e that represents necessary communication between O_v to O_u
    # - - Union({Y_e}_{e \in E_c}) = //// 
    iter = 0
    while iter <= 10:
        iter += 1
        longest_path_length(o_sdag)
        cluster_dag_to_dag(o_sdag, weight_max_cutoff)
        o_sdag = contract_to_coarsened_dag(o_sdag)
        # if make_plots:
        #     draw(o_sdag, title=f'images/contracted_{iter}_O_SDAG')
        if not nx.is_directed_acyclic_graph(o_sdag):
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ NOT DAG @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            raise ValueError('NOT DAG')

    # Problem size has been reduced:
    # check reduced graph
    # check_coarse_graph(o_sdag, sdag, total_weight)
    # weight_max_cutoff = None
    if make_plots:
        draw(o_sdag, title= 'images/COARSE_O_SDAG')
    return o_sdag



def check_coarse_graph(coarse_graph, sdag, total_weight):
    # Checks:
    # Basically check for conservation of variables, operations and weights
    # 1) All intermediate variables should be accounted for:
    # - - Union({v[clustered_variables]}_{v \in V_cg}, {e[edge_variables]}_{e \in E_cg}) = Variables_g/{leaves[V_g]}
    # 2) All operations should be accounted for:
    # - - Union({v[clustered_operiantions]}_{v \in V_cg}) = Operations_g
    # 3) total operation cost is conserved

    # If these checks do not pass, throw an error:
    # 1) Intermediate variables:
    all_intermediate_vars = set()
    for node in sdag.nodes:
        if sdag.nodes[node]['type'] == 'variable':
            if (sdag.in_degree(node) == 0) or (sdag.out_degree(node) == 0):
                continue
            all_intermediate_vars.add(node)

    all_o_sdag_vars = set()
    # Add coarsened edge variables
    for edge in coarse_graph.edges:
        edge_object = coarse_graph.edges[edge]            
        
        # edge nodes of current edge:
        for edge_node in edge_object['edge_variables']:
            all_o_sdag_vars.add(edge_node)
    # Add clustered edge variables
    for node in coarse_graph.nodes:
        node_object = coarse_graph.nodes[node]

        for clustered_var in node_object['clustered_variables']:
            all_o_sdag_vars.add(clustered_var)

    # Check:
    diff = all_intermediate_vars.symmetric_difference(all_o_sdag_vars)
    if len(diff) != 0:
        print('VARIABLES UNACCOUNTED FOR:')
        print('DIFF NODES:', diff)
        print('(CONTRACTED NODES:)', all_o_sdag_vars)
        print('(IV NODES:)',all_intermediate_vars)
        raise KeyError('edge nodes not capturing all intermediate variables')
    
    # 2) Operations
    all_operations = set()
    for node in sdag.nodes:
        if sdag.nodes[node]['type'] == 'operation':
            all_operations.add(node)
    all_o_sdag_ops = set()
    computed_time_cost = 0
    # Add clustered edge operations
    for node in coarse_graph.nodes:
        node_object = coarse_graph.nodes[node]

        computed_time_cost += node_object['weight']

        for clustered_op in node_object['operation_cluster']:
            if clustered_op in all_o_sdag_ops:
                raise KeyError('operation repeated in coarse graph')
            all_o_sdag_ops.add(clustered_op)

    # Check:
    diff = all_operations.symmetric_difference(all_o_sdag_ops)
    if len(diff) != 0:
        print('VARIABLES UNACCOUNTED FOR:')
        print('DIFF NODES:', diff)
        print('(CONTRACTED NODES:)', all_o_sdag_ops)
        print('(IV NODES:)',all_operations)
        raise KeyError('edge nodes not capturing all operations')
    
    if abs(computed_time_cost - total_weight)/total_weight > 1e-10:
        raise ValueError(f'Time cost not preserved: computed cost ({computed_time_cost}) != total cost({total_weight})')
    
def process_coarse_graph(coarse_graph, o_sdag):

    rename_mapping = {}
    for node in coarse_graph.nodes:
        coarse_graph.nodes[node]['cost'] = coarse_graph.nodes[node]['weight']
        rename_mapping[node] = f'cluster_{node}'

        operation_cluster = coarse_graph.nodes[node]['operation_cluster']
        coarse_graph.nodes[node]['operation_subgraph'] = o_sdag.subgraph(operation_cluster)

    nx.relabel_nodes(coarse_graph, rename_mapping, copy=False)