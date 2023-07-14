import matplotlib.pyplot as plt
import networkx as nx

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

def predict_time_temp(rep, manual_wait_time = 0.005):
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
    for node in graph:
       if isinstance (node, OperationNode):
            if isinstance(node.op, StandardOperation):
                # print('OP')
                time = 1e-5
            elif isinstance(node.op, CustomExplicitOperation):
                time = 1e-4
            else:
                # print('NOT OP')
                time = 1e-1
            node.execution_time = time
