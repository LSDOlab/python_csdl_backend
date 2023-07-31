import matplotlib.pyplot as plt
import networkx as nx
from csdl.operations.sparsematmat import sparsematmat
def draw(G,label_attribute = None, title = 'SDAG'):
    plt.figure()
    plt.title(title)
    color_map = []
    for node in G.nodes:
        if 'type' in G.nodes[node]:
            if G.nodes[node]['type'] == 'variable':
                color_map.append('orange')
            else:
                color_map.append('skyblue')
        else:
            color_map.append('mediumseagreen')
            
    pos = nx.kamada_kawai_layout(G)

    if label_attribute:

        labels = nx.get_node_attributes(G, label_attribute) 

        if len(labels) == 0:
            raise KeyError('attribute does not exist???')
        # print(labels)
        nx.draw(G,node_color = color_map, labels=labels, with_labels=True, pos = pos)
    else:
        nx.draw(G,node_color = color_map, with_labels=True,pos = pos)
    plt.savefig(title)
    plt.show()
    # exit(title)


def time_parallel(model, comm,algorithm ,numruns = 10, outputs = [], save = None):
    
    o = outputs
    from mpi4py import MPI
    from time_prediction_v5.time_prediction.predict_time import predict_time
    import python_csdl_backend as pcb
    import csdl
    import time

    m = model
    rep = csdl.GraphRepresentation(m)
    predict_time(rep)


    sim = pcb.Simulator(rep, comm = comm, display_scripts=0, analytics=0, algorithm = algorithm)
    # profiler.disable()
    # profiler.dump_stats(f'output_{comm.rank}')
    # exit()
    sim.run()

    # import cProfile
    # profiler = cProfile.Profile()
    # profiler.enable()

    if comm.rank == 0:
        s = time.time()
    for _ in range(numruns):
        sim.run()

    # profiler.disable()
    # profiler.dump_stats(f'output_{comm.rank}')
    if comm.rank == 0:
        time_out = (time.time() - s)/numruns
        print('TIME:', time_out)
        for key in o:
            print(key, sim[key])


        if save:
            import pickle
            import os

            filename = f'{save}_results.pkl'
            # Check if the pickle file exists
            if os.path.exists(filename):
            # Load the dictionary from the pickle file
                with open(filename, 'rb') as file:
                    data = pickle.load(file)
            else:
                data = {}

            # Modify the value associated with the key 'i'

            if algorithm not in data:
                data[algorithm] = {}

            data[algorithm][comm.size] = time_out

            # Save the modified dictionary back to the pickle file
            with open(f'{filename}', 'wb') as file:
                pickle.dump(data, file)

    del sim
    del m
    del rep

def plot_parallel():
    import pickle
    import matplotlib.pyplot as plt

    # Load the dictionary from the pickle file
    with open('presults.pkl', 'rb') as file:
        data = pickle.load(file)

    # Extract the keys and values from the dictionary
    keys = list(data.keys())
    values = list(data.values())

    # Create a plot
    plt.bar(keys, values)
    plt.xlabel('Keys')
    plt.ylabel('Values')
    plt.title('Plot of Keys vs Values')
    plt.show()
    


from csdl.rep.operation_node import OperationNode
from csdl.lang.standard_operation import StandardOperation
from csdl.lang.custom_explicit_operation import CustomExplicitOperation

def predict_time_temp(rep,measure_bool, manual_wait_time = 0.005):
    """
    Predicts the time of a CSDL Model by iterating through each VariableNode, and either calling predict_operation_time or predict_manual_time.
    predict_operation_time is called for operations in the standard library that have a saved surrogate model, while predict_manual_time is 
    called for explicit/implicit operations and any other operations that don't have a saved surrogate model. Multiplies by the calibration constant
    value saved in the package to adjust for timing differences between computers. If inaccurate predictions are being produced, try recalibrating 
    the package.

    Parameters:
    -----------
    rep: GraphRepresentation
        The GraphRepresentation of the CSDL Model

    Returns:
    --------
    total_time: float
        The predicted amount of time that the Model will take to execute
    """
    graph = rep.flat_graph
    total_time = 0
    nn = len(graph.nodes)
    for i, node in enumerate(graph):
    #    print(f'{i}/{nn}')
       if isinstance (node, OperationNode):
            if isinstance(node.op, StandardOperation):
                # print('OP')
                # if measure_bool:
                #     time = predict_manual_time(rep, node)
                # else:
                if isinstance(node.op, sparsematmat):
                    time = 1e-2
                else:
                    time = 1e-5
            elif isinstance(node.op, CustomExplicitOperation):
                if measure_bool:
                    time = predict_manual_time(rep, node)
                else:
                    time = 1e-4
            else:
                # print('NOT OP')
                if measure_bool:
                    time = predict_manual_time(rep, node)
                else:
                    time = 1e-1
            node.execution_time = time


def predict_manual_time (rep, node):

    """
    Predicts the execution time of a given node in a computation graph by either retrieving it from a cache file or computing it manually.

    Args:
    rep (GraphRepresentation): A GraphRepresentation of a CSDL Model
    node (object): A node in the computation graph.

    Returns:
    t - float: The predicted execution time of the node.
    """
    import os
    import pandas as pd
    from python_csdl_backend import Simulator
    import gc
    from copy import copy 

    pre_saved = False
    # node_hash = hash (node)
    # node_hash = sha256 (pickle.dumps (node)).hexdigest ()
    # node_hash = repr (node)
    node_hash = str(node.name)
    if (os.path.isfile ('./timing/cache.csv')):
        df = pd.read_csv ("./timing/cache.csv")
        pre_saved = node_hash in list (df['hash'])
    else:
        if not os.path.isdir ('./timing'):
            os.mkdir ('./timing')
        pre_saved=False
        template = {'hash':[],
                    'time':[]}
        df = pd.DataFrame (template)
        df.to_csv ('./timing/cache.csv', index=False, columns=['hash', 'time'])

    if (pre_saved):
        t = df ['time'][list (df['hash']).index (node_hash)]
        # print (t)
        # print ("used cache")
    else:
        # print ("predicting manual time for: ", str(node.op).split ()[0].split('.')[-1])
        rep2 = copy(rep)
        rep2.flat_graph = nx.DiGraph()
        rep2.flat_graph.add_node(node)
        rep2.flat_graph.add_edges_from (rep.flat_graph.in_edges (node))
        rep2.flat_graph.add_edges_from (rep.flat_graph.out_edges (node))

        sim = Simulator(rep2, time_prediction=False)
        sim.run ()

        import time

        s = time.time()
        sim.run ()
        t = time.time() - s
        data = {'hash':[node_hash],
                'time':[t]}
        df = pd.DataFrame(data)
        df.to_csv('./timing/cache.csv', mode='a', index=False, header=False)
        del (rep2)
        del (sim)
        gc.collect()
        print('manual')
    return t