import time
import numpy as np

def run_code(code_object, ccl_graph, RANK, NUM_RUNS, VARIABLE_SIZE, num_coms, num_ops, num_coms_pre):
    sdag = ccl_graph
    print("RANK:   ", RANK, f'STARTING EXECUTION ({num_coms} COMMS/{num_ops} OPS/ ({num_coms_pre} PRECOMMS)))')
    time_total = 0
    # exit()
    for i in range(NUM_RUNS):
        s = time.time()
        if RANK == 0:
            variable_dict = {}
            for node in sdag.nodes:
                if 'output' in sdag.nodes[node]:
                    variable_dict[node] = 'IF YOU SEE THIS, ERROR'
            for node in sdag.nodes:
                if sdag.in_degree(node) == 0:
                    variable_dict[node] = np.ones((VARIABLE_SIZE,))
                    # variable_dict[node] = np.ones((VARIABLE_SIZE,))*CONSTANT_MULT+(i/NUM_RUNS)
        else:
            variable_dict = {}


        exec(code_object, {}, variable_dict)
        time_total += time.time() -s
        if RANK == 0:
            output_final = 10.0
            for node in sdag.nodes:
                if 'output' in sdag.nodes[node]:
                    # print(node, variable_dict[node][0], variable_dict[node][1])
                    output_final += np.linalg.norm(variable_dict[node])
            print('FINAL OUTPUT:', output_final)
    print(f'AVG TIME {NUM_RUNS} RUNS ({RANK}): \t', time_total/NUM_RUNS)

