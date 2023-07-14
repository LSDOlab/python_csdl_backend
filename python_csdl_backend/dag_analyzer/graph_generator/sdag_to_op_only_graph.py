import networkx as nx

def sdag_to_op_only_graph(sdag):
    """
    basically makes an operation-only graph from a csdl-like graph. 
    It deletes operation nodes so its easier to process.

    o_sdag must not contain any variable nodes and must not contain any less operation nodes than sdag
    """

    print('translating SDAG to operation-only sdag')
    # loop through each node in sdag and delete variable nodes
    o_sdag = sdag.copy()
    num_ops = 0
    total_weight = 0
    for node in sdag.nodes:
        node_object = sdag.nodes[node]
        """
        We contract one node at a time: variable `var` to predecessor `op_p`

        CASE 1: One output
        BEFORE `var` contraction:
               op_p      var    op_s  
        o -----> o -----> o -----> o 
                            |----> o
        
        AFTER `var` contraction:
               op_p              op_s  
        o -----> o ----- var ----> o 
                   |---- var ----> o


        CASE 2: Multiple outputs
        BEFORE `var` contraction:
               op_p      var     op_s  
        o -----> o -----> o -----> o
                   |
                   |     var2    op_s_2
                   |----> o -----> o
        
        AFTER `var` contraction:
               op_p              op_s  
        o -----> o ----- var ----> o
                   |
                   |    var2    op_s_2
                   |----> o -----> o


        CASE 3: Multiple outputs back to one
        BEFORE `var` contraction:
               op_p      var     op_s  
        o -----> o -----> o -----> o
                   |               ^
                   |     var2      | 
                   |----> o -------|
        
        AFTER `var` contraction:
               op_p              op_s  
        o -----> o ----- var ----> o
                   |               ^
                   |     var2      | 
                   |----> o -------|

        (AFTER 'var2' contraction:)
                op_p                   op_s  
        o -----> o ----- var, var2 ----> o

        """

        if node_object['type'] == 'variable':
            if sdag.in_degree(node) == 1:
                variable_op_source = list(sdag.predecessors(node))[0]
                nx.contracted_nodes(
                    G = o_sdag,
                    u = variable_op_source, # u is kept 
                    v = node, # v is merged into u and deleted
                    self_loops = False,
                    copy = False,
                )

                for successor in sdag.successors(node):
                    if 'edge_variables' not in o_sdag.edges[(variable_op_source, successor)]:
                        o_sdag.edges[(variable_op_source, successor)]['edge_variables'] = set()    
                    o_sdag.edges[(variable_op_source, successor)]['edge_variables'].add(node)
            else:
                o_sdag.remove_node(node)
        else:
            # o_sdag.nodes[node]['time'] = 1.5
            total_weight += o_sdag.nodes[node]['time_cost']
            o_sdag.nodes[node]['weight'] = o_sdag.nodes[node]['time_cost']
            num_ops += 1
    
    # check for correctness
    num_nodes_osdag = 0
    for operation in o_sdag:
        num_nodes_osdag += 1
        node_object = o_sdag.nodes[operation]
        node_object['operation_cluster'] = {operation}
        node_object['clustered_variables'] = set()
        if node_object['type'] != 'operation':
            raise ValueError(f'{operation} is not an operation')        

    # Collect all intermediate variables in sdag.
    # The edges in o_sdag should represent all intermediate variables.
    # that is, *** Union({e['edge_var']}_{e \in o_sdag edges}) == V_sdag
    all_intermediate_vars = set()
    for node in sdag.nodes:
        if sdag.nodes[node]['type'] == 'variable':
            if (sdag.in_degree(node) == 0) or (sdag.out_degree(node) == 0):
                continue
            all_intermediate_vars.add(node)

    # check for *** property:
    all_o_sdag_vars = set()
    for edge in o_sdag.edges:
        edge_object = o_sdag.edges[edge]
        if 'edge_variables' not in edge_object:
             raise KeyError('edge nodes attribute not found')
            
        # edge nodes of current edge:
        for edge_node in edge_object['edge_variables']:
            all_o_sdag_vars.add(edge_node)

    
    diff = all_intermediate_vars.symmetric_difference(all_o_sdag_vars)
    if len(diff) != 0:
        print('VARIABLES UNACCOUNTED FOR:')
        print('DIFF NODES:',diff)
        print('(EDGE NODES:)', all_o_sdag_vars)
        print('(IV NODES:)',all_intermediate_vars)
        raise KeyError('edge nodes not capturing all intermediate variables')
    
    # Check to make sure all operations are correct
    if num_ops != num_nodes_osdag:
        raise ValueError(f'number of nodes in o sdag ({num_nodes_osdag}) != number of nodes in sdag ({num_ops})')

    if not nx.is_directed_acyclic_graph(o_sdag):
        raise ValueError('not a dag')

    return o_sdag, total_weight
