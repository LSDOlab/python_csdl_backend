def name_to_int(node_name):
    return int(node_name[1:])

def code_generator(schedule, rank_graph, VARIABLE_SIZE, RANK, save = 0, sleep = True):

    generated_code_string = 'import numpy as np \n'
    generated_code_string = ''

    if 0:
        generated_code_string += 'import os \n'
        nthreads = 1
        generated_code_string +=  f'os.environ["OMP_NUM_THREADS"] = \'{str(nthreads)}\'\n'
        generated_code_string +=  f'os.environ["OPENBLAS_NUM_THREADS"] = \'{str(nthreads)}\'\n'
        generated_code_string +=  f'os.environ["MKL_NUM_THREADS"] = \'{str(nthreads)}\'\n'
    generated_code_string += 'import numpy as np \n'
    generated_code_string += 'from mpi4py import MPI \n'
    generated_code_string += 'comm = MPI.COMM_WORLD \n'
    generated_code_string += 'import time \n'
    num_ops = 0
    num_coms = 0
    num_coms_pre = 0
    before_first_op = True

    for node in schedule:

        # Write code for every node for a partricular rank
        # If send/recieve, we need to either recieve or send the information
        # if ('SEND' in node):
        #     generated_code_string += generate_send(node, VARIABLE_SIZE)
        #     num_coms+=1
        # elif ('GET' in node):
        #     generated_code_string += generate_receive(node, VARIABLE_SIZE)
        #     num_coms+=1
        # elif ('IRECV' in node):
        #     generated_code_string += generate_receive(node, VARIABLE_SIZE, with_wait = True)
        #     num_coms+=1
        # elif ('SsENDONLY' in node):
        #     generated_code_string += generate_send(node, VARIABLE_SIZE)
        #     num_coms+=1
        # elif ('irecvwait' in node):
        #     generated_code_string += generate_receive_wait(node)
        # elif ('WAITING' in node) or ('WF' in node):
        #     continue

        string, num_coms = generate_mpi_operators(node, (VARIABLE_SIZE, ), num_coms)
        
        if string == 'continue':
            continue
        elif string is not None:
            generated_code_string += string
        else:
            if before_first_op:
                num_coms_pre = num_coms
            before_first_op = False
            
        # If operation node, simply write the operation. No need to worry about send and recieve
            node_object = rank_graph.nodes[node]
            if node_object['type'] == 'operation':

                if sleep:
                    op_string = generate_operation(
                        list(rank_graph.predecessors(node)), 
                        list(rank_graph.successors(node)), 
                        node, 
                        sleep = rank_graph.nodes[node]['time_cost'])
                    generated_code_string += op_string
                else:
                    op_string = generate_operation(
                        list(rank_graph.predecessors(node)), 
                        list(rank_graph.successors(node)), 
                        node, )
                    generated_code_string += op_string

                num_ops+=1
        # else:
        #     if 'rank_sources' in node_object:
        #         if node_object['rank_sources'] != {rank}:
        #             generated_code_string += generate_receive(node, node_object)
        #             num_coms+=1
        #     if 'rank_targets' in node_object:
        #         generated_code_string += generate_send(node, node_object)
        #         num_coms+=1
        
        # # if rank == 0:
        # if 1:
        #     if rank > -1:
        #         if node_object['type'] == 'variable':
        #             generated_code_string += f'print({rank},\'{node}\',{node})\n'
                    # generated_code_string += f'print({rank},\'{node}\')\n'
    generated_code_string += 'comm.barrier()\n'

    import os

    # Set the name of the directory
    directory = "CODE"

    # Check if the directory exists
    if not os.path.exists(directory):
        # If the directory doesn't exist, create it
        os.makedirs(directory)

    code_name = f'{directory}/rank_{RANK}_code.py'

    # Open the file for writing
    if save:
        with open(code_name, "w") as f:
            # Write the string to the file
            f.write(generated_code_string)

    code_object = compile(generated_code_string, code_name, 'exec')

    return code_object, num_coms, num_ops, num_coms_pre


def generate_mpi_operators(node, shape, num_coms, tag = None):
    args_list = [node, shape]
    if tag is not None:
        args_list.append(tag)
    if ('SEND' in node):
        operator_string = generate_send(*args_list)
        num_coms+=1
    elif ('GET' in node):
        operator_string = generate_receive(*args_list)
        num_coms+=1
    elif ('IRECV' in node):
        operator_string = generate_receive(*args_list, with_wait = True)
        num_coms+=1
    elif ('SsENDONLY' in node):
        operator_string = generate_isend(*args_list)
        num_coms+=1
    elif ('irecvwait' in node):
        operator_string = generate_receive_wait(node)
    elif ('WAITING' in node) or ('W/F' in node):
        return 'continue',num_coms
    else:
        return None, num_coms
    return operator_string, num_coms

def generate_operation(inputs, outputs, operation, sleep = None):
    string = f"\n# OPERATION {operation}\n"
    for i, input_var in enumerate(inputs):
        if i == 0:
            string += f"temp = ({input_var})\n"
        else:
            string += f"temp = ({input_var})*temp\n"
            string += f"temp = ({input_var})\n"
    
    # for i in range(1000):
    if sleep is not None:
        string+= f"time.sleep({sleep})\n"
        # print(sleep)
        # string+= f"time.sleep(0.0000000001)\n"
        # pass
    else:
        # string+= "time.sleep(0.001)\n"
        for i in range(00):
            # string += f"x = (temp+temp+temp+temp+temp+temp)+(temp+temp+temp+temp+temp+temp+0.01)\n"
            string += f"x = np.sin(temp)/(np.sin(temp)+0.001)\n"
            # string += f"x = temp*np.dot(1/temp, temp)\n"
            # string += f"x = temp*temp*temp*temp*temp*temp\n"

    for i, output_var in enumerate(outputs):
        string += f"{output_var} = temp*(1/{i+1})\n"
        string += f"{output_var} = temp*(1.0000{i+1})\n"
        string += f"{output_var} = temp\n"


    return string

def generate_receive(send_string, VARIABLE_SIZE,  node_int = (None,None), with_wait = False): 
    # string = f"\n# RECIEVE {node}\n"

    # sources = '['
    # for i, source_rank in enumerate(node_object['rank_sources']):
    #     sources += f'{source_rank},'
    # sources += ']'
    # # exit()
    
    # sources = list(node_object['rank_sources'])[0]
    # sources = 'MPI.ANY_SOURCE'

    # tag_int = name_to_int(node)
    # # string += f"{node} = np.empty(({VARIABLE_SIZE},))\n"
    # string += f"{node} = 100000*np.ones(({VARIABLE_SIZE},))\n"
    # string += f"comm.Recv({node}, source={sources}, tag = {tag_int})\n"
    # string += f"comm.Irecv({node}, source={sources}, tag = {tag_int})\n"
    # string += f"{node} = comm.recv(source={sources}, tag = {tag_int})\n"

    split_string = send_string.split("/")
    if node_int == (None,None):
        var_to_recv= split_string[1]
        tag_int = name_to_int(var_to_recv)
    else:
        var_to_recv = node_int[0]
        tag_int = node_int[1]
    source_rank = split_string[3]
    string = f"\n# RECV {var_to_recv}\n"
    if isinstance(VARIABLE_SIZE, int):
        string += f"{var_to_recv} = np.ones(({VARIABLE_SIZE},))\n"
    else:
        string += f"{var_to_recv} = np.ones({VARIABLE_SIZE})\n"

    if not with_wait:
        string += f'comm.Recv({var_to_recv}, source = {source_rank}, tag = {tag_int})\n'
    else:
        string += f'wait_{var_to_recv} = comm.Irecv({var_to_recv}, source = {source_rank}, tag = {tag_int})\n'

    # string += f'comm.Irecv({var_to_recv}, source = {source_rank}, tag = {tag_int})\n'

    # print(string)
    return string

def generate_receive_wait(send_string):
    split_string = send_string.split("/")
    var_to_wait = split_string[1]
    string = ""
    # string = f'print(\'wait_{var_to_wait}.wait()\')\n'
    string += f'wait_{var_to_wait}.wait()\n'
    # string += f'print(\'wait_{var_to_wait}.wait() over\')\n'
    # string += f'comm.barrier()\n'
    if var_to_wait == 'v0679_total_torque':
        string += f'print(\'{var_to_wait}\', {var_to_wait})\n'
    return string


def generate_send(send_string, VARIABLE_SIZE, node_int = (None,None)):
    # string = f"\n# SEND {node}\n"
    # tag_int = name_to_int(node)
    # for i, target_rank in enumerate(node_object['rank_targets']):
        # string += f'comm.Isend({node}, dest = {target_rank}, tag = {tag_int})\n'
        # string += f'comm.Send({node}, dest = {target_rank}, tag = {tag_int})\n'
        # string += f'comm.send({node}, dest = {target_rank}, tag = {tag_int})\n'

    # string = f''

    split_string = send_string.split("/")
    if node_int == (None,None):
        var_to_send = split_string[1]
        tag_int = name_to_int(var_to_send)
    else:
        var_to_send = node_int[0]
        tag_int = node_int[1]
    target_rank = split_string[3]
    string = f"\n# SEND {var_to_send}"
    string += f"\n# SHAPE {VARIABLE_SIZE}\n"
    string += f'comm.Send({var_to_send}, dest = {target_rank}, tag = {tag_int})\n'
    # string += f'comm.Isend({var_to_send}, dest = {target_rank}, tag = {tag_int})\n'

    # string += f'comm.Isend({var_to_send}, dest = {target_rank}, tag = {tag_int})\n'
    return string

def generate_isend(send_string, VARIABLE_SIZE, node_int = (None,None)):
    # string = f"\n# SEND {node}\n"
    # tag_int = name_to_int(node)
    # for i, target_rank in enumerate(node_object['rank_targets']):
        # string += f'comm.Isend({node}, dest = {target_rank}, tag = {tag_int})\n'
        # string += f'comm.Send({node}, dest = {target_rank}, tag = {tag_int})\n'
        # string += f'comm.send({node}, dest = {target_rank}, tag = {tag_int})\n'

    # string = f''

    split_string = send_string.split("/")
    if node_int == (None,None):
        var_to_send = split_string[1]
        tag_int = name_to_int(var_to_send)
    else:
        var_to_send = node_int[0]
        tag_int = node_int[1]
    target_rank = split_string[3]
    string = f"\n# SEND {var_to_send}"
    string += f"\n# SHAPE {VARIABLE_SIZE}\n"
    # string += f'print(\'{var_to_send}.send()\')\n'

    # OLD
    string += f'comm.Isend({var_to_send}, dest = {target_rank}, tag = {tag_int})\n'
    
    # NEW: seems to work but a bit slower
    # string += f'x = comm.Issend({var_to_send}, dest = {target_rank}, tag = {tag_int})\n'
    # string += f'x.wait()\n'

    # Other: tried other options but doesn't work that well
    # string += f'x.Free()\n'
    # print('lskdfm')
    # string += f'comm.Isend({var_to_send}, dest = {target_rank}, tag = {tag_int})\n'
    # string += f'x.wait()\n'
    # string += f'print(\'{var_to_send}.send over\')\n'

    return string