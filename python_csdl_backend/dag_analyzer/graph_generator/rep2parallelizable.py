import random
import pickle
import networkx as nx
import numpy as np

try:
    from csdl.rep.operation_node import OperationNode
    from csdl.rep.variable_node import VariableNode
except:
    pass


def rep2parallelizable(
        rep):
    
    str_to_node = {}
    
    # if comm.rank == 0:
    #     # print('KJSDFKNSDSKNFKSDJFSDKNFSDNFKJSDNFKJDSNFKJDSNFJKSDNFJKSDNFJKSDN')
    #     parallelizable_graph = nx.DiGraph()

    #     for (u,v) in rep.flat_graph.edges:
            
    #         node_pred = rep2p_node(u)
    #         node_succ = rep2p_node(v)

    #         str_to_node[node_pred[0]] = u
    #         str_to_node[node_succ[0]] = v

    #         parallelizable_graph.add_edge(node_pred[0], node_succ[0])
    #         parallelizable_graph.nodes[node_pred[0]]['type'] = node_pred[1]
    #         parallelizable_graph.nodes[node_succ[0]]['type'] = node_succ[1]
    # else:
    #     parallelizable_graph = None

    # parallelizable_graph = comm.bcast(parallelizable_graph, root = 0)

    parallelizable_graph = nx.DiGraph()

    for (u,v) in rep.flat_graph.edges:
        
        node_pred = rep2p_node(u)
        node_succ = rep2p_node(v)

        str_to_node[node_pred[0]] = u
        str_to_node[node_succ[0]] = v

        parallelizable_graph.add_edge(node_pred[0], node_succ[0])
        parallelizable_graph.nodes[node_pred[0]]['type'] = node_pred[1]
        parallelizable_graph.nodes[node_succ[0]]['type'] = node_succ[1]

        parallelizable_graph.nodes[node_pred[0]]['cost'] = node_pred[2]
        parallelizable_graph.nodes[node_succ[0]]['cost'] = node_succ[2]

        if node_pred[1] == 'operation':
            parallelizable_graph.nodes[node_pred[0]]['time_cost'] = node_pred[2]
        else:
            parallelizable_graph.nodes[node_pred[0]]['memory_cost'] = node_pred[3]
        if node_succ[1] == 'operation':
            parallelizable_graph.nodes[node_succ[0]]['time_cost'] = node_succ[2]

    for u in rep.flat_graph.nodes:
        node_pred = rep2p_node(u)

        str_to_node[node_pred[0]] = u

        parallelizable_graph.add_node(node_pred[0])
        parallelizable_graph.nodes[node_pred[0]]['type'] = node_pred[1]
        parallelizable_graph.nodes[node_pred[0]]['cost'] = node_pred[2]
        if node_pred[1] == 'operation':
            parallelizable_graph.nodes[node_pred[0]]['time_cost'] = node_pred[2]
        else:
            parallelizable_graph.nodes[node_pred[0]]['memory_cost'] = node_pred[3]

    # exit()
    return parallelizable_graph, str_to_node
    
def rep2p_node(node):
    if isinstance(node, VariableNode):
        node_string = node.id
        node_type = 'variable'
        node_cost = 1e-4
        
        # memory cost is 4 bytes per element????
        memory_cost = (np.prod(node.var.shape))*4
    elif isinstance(node, OperationNode):
        node_string = node.name
        node_type = 'operation'
        node_cost = float(node.execution_time)
        if node_cost < 0.0:
            node_cost = 1e-6
            # raise ValueError('NEGATIVE COST', node_cost, node_string)
        if node_cost == 0.10:
            pass
            # print(node.op, node_string, node_cost)
        memory_cost = None
    else:
        raise KeyError('what is this')
    return node_string, node_type, node_cost, memory_cost
            