import random
import pickle
import networkx as nx
import numpy as np
# from python_csdl_backend.dag_analyzer.schedulers.scheduler import generate_schedule_plot
from python_csdl_backend.dag_analyzer.utils import draw

iter_name = 0
def create_name(type_node):
    global iter_name
    iter_name +=1
    # print(iter_name)
    return type_node[0]+str(iter_name)

def create_csdl_like_graph(
        comm,
        graph_size,
        super_output,
        load_graph: str = None,
        structure = 'random',
        num_paths:int  = 5,
        save = None,
    ):
    if comm.rank == 0:
        n = graph_size
        if load_graph:
            with open(f'{load_graph}.pickle', 'rb') as f:
                G =  pickle.load(f)
        else:
            print(f'Creating graph of size {n}')
            G = nx.DiGraph()

            if 'embarassing' in structure:
                num_in_chain = max(round(n/num_paths), 1)
                num_in_chain = int(np.ceil(num_in_chain / 2.) * 2)

                name_int = 0
                root_node_name = create_name('variable')
                G.add_node(root_node_name, type='variable')
                to_last_node = set()
                for _ in range(num_paths):
                    
                    if 'diamond' in structure:
                        # node_type = 
                        node_name = create_name('operation')
                        G.add_node(node_name, type='operation')
                        G.nodes[node_name]['time_cost'] = 500
                        G.add_edge(root_node_name, node_name)


                        # node_name_almost_last = create_name('operation')
                        # G.add_node(node_name_almost_last, type='operation')
                        # to_last_node.add(node_name_almost_last)

                        G2 = create_csdl_like_graph(
                            comm,
                            num_in_chain,
                            super_output=1,
                            structure='embarassing',
                            num_paths=2,
                        )
                        # draw(G2, title='TEST')


                        for node2 in G2.nodes:
                            G.add_node(node2, type = G2.nodes[node2]['type'])

                            if G2.nodes[node2]['type'] == 'operation':
                                G.nodes[node2]['time_cost']= G2.nodes[node2]['time_cost']

                            if G2.in_degree(node2) == 0:
                                G.add_edge(node_name, node2)
                            if G2.out_degree(node2) == 0:
                                to_last_node.add(node2)
                                # G.add_edge(node2, node_name_almost_last)

                        for u,v in G2.edges:
                            G.add_edge(u, v)
                        # print(G)
                    else:
                        for j in range(num_in_chain):
                            if j == 0:
                                node_type = 'operation'
                                node_name = create_name(node_type)
                                G.add_node(node_name, type=node_type)
                                G.nodes[node_name]['time_cost'] = 500
                                G.add_edge(root_node_name, node_name)
                                prev_name = node_name
                                continue
                            
                            # even == operations
                            if (j % 2 == 0):
                                node_type = 'operation'
                            else:
                                node_type = 'variable'

                            # print(j, node_type)
                            node_name = create_name(node_type)
                            G.add_node(node_name, type=node_type)
                            G.add_edge(prev_name, node_name)
                            prev_name = node_name

                            if node_type == 'operation':
                                G.nodes[node_name]['time_cost'] = 500

                            if j == num_in_chain-1:
                                to_last_node.add(node_name)

                name_int += 1
                final_node_name = create_name('operation')
                G.add_node(final_node_name, type='operation')
                G.nodes[final_node_name]['time_cost'] = 500
                for prev in to_last_node:
                    G.add_edge(prev, final_node_name)

                name_int += 1
                real_final_node_name = create_name('variable')
                G.add_node(real_final_node_name, type='variable')
                G.add_edge(final_node_name, real_final_node_name)


            elif structure == 'random':
                # Create nodes
                for i in range(n):
                    node_type = 'variable' if random.random() < 0.5 else 'operation'
                    node_name = create_name(node_type)

                    G.add_node(node_name, type=node_type)
                # Create edges
                variable_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'variable']
                operation_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'operation']

                # print(variable_nodes)
                # print(operation_nodes)

                for v in variable_nodes:
                    possible_successors = operation_nodes
                    num_successors = random.randint(0, min(10, len(possible_successors)))
                    num_successors = random.randint(0, 1)
                    successors = random.sample(possible_successors, num_successors)
                    G.add_edges_from([(v, s) for s in successors])


                for o in operation_nodes:
                    # possible_successors = variable_nodes.copy()
                    possible_successors = variable_nodes
                    num_successors = random.randint(1, min(10, len(possible_successors)))
                    for i in range(num_successors):
                        successor = random.sample(possible_successors, 1)
                        if len(list(G.predecessors(successor[0]))) == 0:
                            G.add_edges_from([(o, s) for s in successor])

                    if len(list(G.predecessors(o))) == 0:
                        # possible_preds = variable_nodes.copy()
                        possible_preds = variable_nodes

                        pred = random.sample(possible_preds, 1)
                        G.add_edges_from([(pred[0], o)])

                acyclical = nx.is_directed_acyclic_graph(G)
                while not acyclical:
                    print('removing cycle')
                    cycle = nx.find_cycle(G)
                    G.remove_edges_from(cycle)
                    acyclical = nx.is_directed_acyclic_graph(G)

                cur_ind = n
                outputs = [x for x in G.nodes() if G.out_degree(x)==0].copy()
                for leaf_out in outputs:
                    # print(leaf_out, G.nodes[leaf_out]['type'])
                    leaf_out_type = G.nodes[leaf_out]['type']
                    if leaf_out_type == 'operation':
                        node_name= create_name('variable')
                        G.add_node(node_name, type='variable')
                        G.add_edge(leaf_out, node_name)
                    cur_ind+= 1

                inputs = [x for x in G.nodes() if G.in_degree(x)==0].copy()
                for leaf_in in inputs:
                    # print(leaf_in, G.nodes[leaf_out]['type'])
                    leaf_in_type = G.nodes[leaf_in]['type']
                    if leaf_in_type == 'operation':
                        node_name= create_name('variable')
                        G.add_node(node_name, type='variable')
                        G.add_edge(node_name, leaf_in)
                    cur_ind+= 1
                
                if super_output:

                    # SUPER OUTPUT
                    outputs = [x for x in G.nodes() if G.out_degree(x)==0].copy()
                    node_name= create_name('operation')
                    for leaf_out in outputs:
                        G.add_node(node_name, type='operation')
                        G.add_edge(leaf_out, node_name)

                    cur_ind+=1
                    node_name2= create_name('variable')
                    G.add_node(node_name2, type='variable')
                    G.add_edge(node_name, node_name2)

                # add `time cost` to each operation
                for node in G.nodes:
                    cur_type = G.nodes[node]['type']
                    if cur_type == 'operation':
                        # G.nodes[node]['time_cost'] = round(random.uniform(0.5, 50.0), 1)
                        G.nodes[node]['time_cost'] = int(round(random.uniform(1, 1000), 6))
                        # G.nodes[node]['time_cost'] = 1000
            else:
                raise KeyError('wrong graph structure')

            # Check to see if the graph G is legit
            # rules:
            # - 1) every leaf node is a variable
            # - 2) all variables can only have one or zero out edges
            # - 3) all nodes must have opposite types as their neighbors

            # draw(G, title='TEST')
            
            for node in G.nodes:
                cur_type = G.nodes[node]['type']

                # 1)
                if len(list(G.predecessors(node))) == 0:
                    if cur_type == 'operation':
                        error_s = f'INPUT {node} is not a variable'
                        raise TypeError(error_s)
                if len(list(G.successors(node))) == 0:
                    if cur_type == 'operation':
                        error_s = f'OUTPUT {node} is not a variable'
                        raise TypeError(error_s)
                    
                # 2)
                if cur_type == 'variable':
                    num_predecessors = len(list(G.predecessors(node)))
                    if num_predecessors not in [0,1]:
                        error_s = f'variable {node} has {num_predecessors} predecessors'
                        raise ValueError(error_s)
                else:
                    if 'time_cost' not in G.nodes[node]:
                        error_s = f'operation {node} has no time cost'
                        raise ValueError(error_s)

                # 3)
                for s in G.successors(node):
                    if G.nodes[s]['type'] == cur_type:
                        error_s = f'variable {node} has wrong successors: {s}'
                        raise ValueError(error_s) 
                for s in G.predecessors(node):
                    if G.nodes[s]['type'] == cur_type:
                        error_s = f'variable {node} has wrong successors: {s}'
                        raise ValueError(error_s) 

                # print(G.nodes[node])
            # save the DiGraph object to a file using pickle
            # print(save)
            if save is not None:
                with open(f'{save}.pickle', 'wb') as f:
                    pickle.dump(G, f)
                    print('skjdfns', save, G)

    else:
        G =None
    
    if comm.size == 1:
    
        print('sldjkfns', G)
        return G
    G = comm.bcast(G, root = 0)

    # create_csdl_like_graph(
    #     comm,
    #     10,
    #     1,
    # )
    return G
