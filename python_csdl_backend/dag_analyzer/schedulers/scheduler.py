import matplotlib.pyplot as plt

class Scheduler():
    
    def __init__(self, algorithm, comm) -> None:
        self.algorithm = algorithm
        if comm is not None:
            self.algorithm.NUM_RANKS = comm.size
        else:
            self.algorithm.NUM_RANKS = 1
        self.comm = comm

    def schedule(
            self,
            ccl_graph,
            profile = 0,
            create_plots = 0,
            visualize_schedule = 0,
            checkpoints = False,
            checkpoint_stride = None,
        ):
        
        if self.comm is not None:
            if self.comm.rank == 0:
                schedule, node_to_owner_rank,checkpoint_data = self.schedule_main_process(
                    ccl_graph,
                    profile,
                    create_plots,
                    visualize_schedule,
                    checkpoints,
                    checkpoint_stride,
                    )
            else:
                schedule = None
                node_to_owner_rank = None
                checkpoint_data = None
            schedule = self.comm.scatter(schedule, root = 0)
            node_to_owner_rank = self.comm.bcast(node_to_owner_rank, root = 0)
            checkpoint_data = self.comm.bcast(checkpoint_data, root = 0)

            return schedule, node_to_owner_rank, checkpoint_data
    
        else:
            schedule, node_to_owner_rank, checkpoint_data = self.schedule_main_process(
                ccl_graph,
                profile,
                create_plots,
                visualize_schedule,
                checkpoints,
                checkpoint_stride,
                )
            return schedule[0], node_to_owner_rank, checkpoint_data


    def schedule_main_process(
            self,
            ccl_graph,
            profile,
            create_plots,
            visualize_schedule,
            checkpoints,
            checkpoint_stride,
        ):

        self.algorithm.set_create_plots(create_plots)

        if profile:
            import cProfile
            profiler = cProfile.Profile()
            profiler.enable()

        schedule, estimated_timeline = self.algorithm.schedule(ccl_graph)
        node_to_owner_rank = check_schedule(schedule, estimated_timeline ,ccl_graph)

        if checkpoints:
        # if 0:
            checkpoint_data = determine_checkpoints(estimated_timeline, schedule, ccl_graph, checkpoint_stride)
        else:
            checkpoint_data = None
        
            
        theoretical_makespan = max([x[-1] for x in estimated_timeline])
        print(f'MAKESPAN ESTIMATION: {theoretical_makespan}')
        if visualize_schedule:
            generate_schedule_plot(estimated_timeline, schedule, checkpoint_data)
            
        if profile:
            profiler.disable()
            profiler.dump_stats('output')

        return schedule, node_to_owner_rank,checkpoint_data


def generate_schedule_plot(estimated_times, full_schedule, checkpoint_data):
    fig, ax = plt.subplots(figsize = (4.5,6))
    colormap = {
        'comm':{ 
            'colors': ['chocolate', 'saddlebrown'],
            'alpha': 1,
            'index': 0
            },
        'wait': {
            'colors': ['white'], 
            'alpha': 0,
            'index': 0
            },
        'operation': {
            'colors': ['lightseagreen', 'mediumturquoise'],
            'alpha': 1,
            'index': 0
            },
    }

    num_ranks = len(estimated_times)

    # look for checkpoints if we have any
    if checkpoint_data is not None:
        checkpoint_plot_data = []
        checkpoint_last_ops = [[None] for _ in range(num_ranks)]
        checkpoint_plot_data.append([-1 for _ in range(num_ranks)])
        for snapshot_index in range(len(checkpoint_data)):
            
            # Dont plot first checkpont
            if snapshot_index == 0:
                continue
            
            checkpoint_plot_data.append([-1 for _ in range(num_ranks)])
            snapshot = checkpoint_data[snapshot_index]
            for rank in range(num_ranks):
                if len(snapshot['snapshot schedule'][rank]) == 0:
                    checkpoint_last_ops[rank].append(checkpoint_last_ops[rank][-1])
                else:
                    checkpoint_last_ops[rank].append(snapshot['snapshot schedule'][rank][-1])
    # for rank in checkpoint_last_ops:
    #     print([rank])

    for rank in range(len(estimated_times)):
        values = estimated_times[rank]
        schedule = full_schedule[rank]
        # Calculate the differences between consecutive values
        heights = [values[i+1] - values[i] for i in range(len(values)-1)]
        bottoms = [values[i] for i in range(len(values)-1)]

        # Initialize the bottom of the bars to be zero
        # bottoms = [0] * len(values)

        # Draw each stack of bars
        for i in range(len(heights)):
            height = heights[i]
            
            if ('GET_/' in schedule[i]):
                operation_type = 'comm'
            elif ('SEND_/' in schedule[i]):
                operation_type = 'comm'
                arrow_y = bottoms[i] + (height)/2
                split_string = schedule[i].split("/")
                target_rank = int(split_string[3])

                if target_rank > rank:
                    x_start = rank + 0.45
                    x_end = target_rank - 0.45
                else:
                    x_start = rank - 0.45
                    x_end = target_rank + 0.45

                ax.arrow(
                    x_start,
                    arrow_y,
                    x_end - x_start,
                    0,
                    width = height/15,
                    head_width = height/3,
                    head_length = 0.03,
                    color = 'black',
                    length_includes_head = True,
                )
            elif ('IRECV' in schedule[i]) or  ('SsENDONLY' in schedule[i]) or ('irecvwait' in schedule[i]):
                operation_type = 'comm'
            elif ('WAIT' in schedule[i]) or ('W/F' in schedule[i]):
                operation_type = 'wait'
            else:
                operation_type = 'operation'
            # print(schedule[i], height)
            alpha = colormap[operation_type]['alpha']
            color_list = colormap[operation_type]['colors']
            color_ind = colormap[operation_type]['index']%len(color_list)
            colormap[operation_type]['index']+=1
            ax.bar(rank, height, bottom=bottoms[i], color=color_list[color_ind], alpha = alpha)

            if checkpoint_data is not None:
                if schedule[i] in set(checkpoint_last_ops[rank]):
                    indices = []
                    for index, e in enumerate(checkpoint_last_ops[rank]):
                        if e == schedule[i]:
                            indices.append(index)

                    for index in indices:
                        checkpoint_plot_data[index][rank] = bottoms[i] + (height)
    
    if checkpoint_data is not None:
        def plot_steps(coordinates):
            for i in range(len(coordinates)):
                x1 = coordinates[i][0] - 0.5
                x2 = coordinates[i][0] + 0.5
                y = coordinates[i][1]
                ax.hlines(y, x1, x2, colors = 'red', linewidths = 0.5)
                if i == len(coordinates) - 1:
                    continue

                ax.vlines(x2, coordinates[i][1], coordinates[i+1][1], colors = 'red', linewidths = 0.5)
        
        for i, current_snapshot_times in enumerate(checkpoint_plot_data):

            coords = []
            for rank, time in enumerate(current_snapshot_times):
                if current_snapshot_times[rank] == -1:
                    end_time = 0
                else:
                    end_time = current_snapshot_times[rank]
                coords.append([rank, end_time])
            # print(coords)
            if i != 0:
                plot_steps(coords)


        # ax.set_ylim(top = 0.17)
        # ax.set_xlim([-0.75, 7.75])
        ax.set_title(f'Static Gantt Chart ({num_ranks} processor(s))')
        ax.set_xlabel(f'Processor Index (Rank)')
        ax.set_ylabel(f'Point in Time')
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='lightseagreen', label='Running Operation')
        red_patch2 = mpatches.Patch(color='chocolate', label='Running MPI Send')
        plt.legend(handles=[red_patch, red_patch2])

            # bottoms[i+1:] = [b+height for b in bottoms[i+1:]]
        # exit('ksjdndkg')
    # Set the x-axis ticks and labels
    # ax.set_xticks([])
    # ax.set_xticklabels([f'{v:.2f}' for v in values])

    plt.savefig(f'schedule_{num_ranks}.png', dpi = 300)
    # plt.savefig(f'schedule_{num_ranks}.pdf', dpi = 300)
    # exit()
    # plt.show()

def check_schedule(all_schedules, estimated_timeline, sdag):

    operations = set()
    node_to_owner_rank = {}
    num_vars = 0
    for node in sdag.nodes:
        if sdag.nodes[node]['type'] == 'operation':
            operations.add(node)
        elif sdag.nodes[node]['type'] == 'variable':
            num_vars += 1
            if sdag.in_degree(node) == 0:
                node_to_owner_rank[node] = {0}
            elif sdag.out_degree(node) == 0:
                node_to_owner_rank[node] =  {0}

    num_sends = 0
    num_recvs = 0
    for rank in range(len(all_schedules)):
        communication_ops = set()
        schedule = all_schedules[rank]
        schedule_timeline = estimated_timeline[rank]
        if len(schedule) != len(schedule_timeline)-1:
            raise ValueError('Schedule timeline inconsistent with schedule')
        # print(f'RANK {rank}:')
        # print(f'\t length: {len(schedule)}')
        # print(f'\t length: {len(schedule)}')

        for operation in schedule:
            if ('SEND_/' in operation) or ('GET_/' in operation) or ('WAITING_for' in operation):
                
                if operation in communication_ops:
                    raise ValueError(f'operation {operation} already in communication_ops')
                
                communication_ops.add(operation)

                # print(operation)
                if 'SEND' in operation:
                    num_sends += 1
                elif 'GET_/' in operation:
                    num_recvs += 1

                continue
            
            if operation not in sdag.nodes:
                raise KeyError(f'operation {operation} not in DAG.')
            
            operations.remove(operation)
        
            for successor in sdag.successors(operation):
                if successor not in node_to_owner_rank:
                    node_to_owner_rank[successor] = set()
                node_to_owner_rank[successor].add(rank)
            for predecessor in sdag.predecessors(operation):
                if predecessor not in node_to_owner_rank:
                    node_to_owner_rank[predecessor] = set()
                node_to_owner_rank[predecessor].add(rank)


    if num_sends != num_recvs:
        raise ValueError(f'number of sends ({num_sends}) does not match number of recvs ({num_recvs})')

    num_ops_left = len(operations)
    if num_ops_left != 0:
        print(operations)
        raise ValueError(f'schedule does not have all operations ( {num_ops_left} left)')
    
    if len(node_to_owner_rank) != num_vars:
        raise ValueError(f'node_to_owner_rank mapping does not have all variables ({num_vars - len(node_to_owner_rank)} left)')

    # exit()
    return node_to_owner_rank
    # exit()

def determine_checkpoints(estimated_timeline, schedules, graph, checkpoint_stride):
    import numpy as np
    from math import ceil

    all_input_vars = set()
    total_mem = 0
    for node in graph:
        if graph.in_degree(node) == 0:
            all_input_vars.add(node)

        if graph.nodes[node]['type'] == 'variable':
            # exit()
            total_mem += graph.nodes[node]['memory_cost']
    input_vars_cost = sum([graph.nodes[node]['memory_cost'] for node in all_input_vars])

    if checkpoint_stride is None:
        max_stride = total_mem/10
        # max_stride = 150_000_000
    else:   
        max_stride = checkpoint_stride # randomly set 1gb stride
    # max_stride = 150 # randomly set 1gb stride
    stride_minimum_ratio = 0.8
    min_stride = max_stride*stride_minimum_ratio


    # Preprocessing
    task_lists = []
    all_scheduled_ops = set()

    total_sends = 0
    total_recvs = 0
    for rank in range(len(schedules)):
        task_list = []
        for i, operation in enumerate(schedules[rank]):
            if 'WAIT' in operation:
                continue
            if 'SEND_/' in operation:
                total_sends += 1
            elif 'GET_/' in operation:
                total_recvs += 1
            all_scheduled_ops.add(operation)
            task_list.append((operation, estimated_timeline[rank][i], estimated_timeline[rank][i+1]))
        task_lists.append(task_list)
    number_of_ranks = len(task_lists)
    rev_tasks_list = [list(reversed(tasks.copy())) for tasks in task_lists]

    # Store the latest task in each list that ends before 'et'
    latest_tasks = [(None, 0, 0)] * number_of_ranks
    zero_dict = {
        'latest tasks': latest_tasks,
        'maximum prefix memory allocation': input_vars_cost,
        'load balance': 0.0,
    }
    mapped_end_times = {0: zero_dict}
    ordered_end_times = [0]

    # Find checkpoints
    while any(len(rev_tasks) > 0 for rev_tasks in rev_tasks_list):
        # Get the end time of the next task in each list
        end_times = []
        for rev_tasks in rev_tasks_list:
            if len(rev_tasks) > 0:
                # end_times.append(rev_tasks[-1][2])
                end_times.append(ceil(rev_tasks[-1][2]*1e6)/1e6)
            else:
                end_times.append(np.inf)
        
        # Find the minimum end time among all lists
        minimum_end_time, minimum_time_rank = min((val, idx) for (idx, val) in enumerate(end_times))
        memory_cost_this_node = 0
        minimum_end_time_node = rev_tasks_list[minimum_time_rank][-1][0]
        if minimum_end_time_node in graph.nodes:
            for outputs in graph.successors(minimum_end_time_node):
                memory_cost_this_node += graph.nodes[outputs]['memory_cost']
        
        # Update the latest task for each list if its end time is less than the minimum end time
        latest_tasks = [(None, None, None)] * number_of_ranks
        for rank, tasks in enumerate(rev_tasks_list):
            if rank == minimum_time_rank:
                latest_tasks[rank] = rev_tasks_list[rank].pop()
            else:
                latest_tasks[rank] = mapped_end_times[ordered_end_times[-1]]['latest tasks'][rank]

        # Add the latest tasks for the minimum end time to the result list
        # mapped_end_times.append((min_end_time, latest_tasks.copy()))
        new_memory = mapped_end_times[ordered_end_times[-1]]['maximum prefix memory allocation'] + memory_cost_this_node

        if minimum_end_time not in mapped_end_times:
            ordered_end_times.append(minimum_end_time)
        mapped_end_times[minimum_end_time] = {}
        mapped_end_times[minimum_end_time]['latest tasks'] = latest_tasks
        mapped_end_times[minimum_end_time]['maximum prefix memory allocation'] = new_memory

        # print(minimum_end_time, mapped_end_times[minimum_end_time]['maximum prefix memory allocation']/100000)

    if len(ordered_end_times) != len(mapped_end_times):
        raise ValueError('ordered_end_times and mapped_end_times have different lengths')
    # print(len(ordered_end_times), len(mapped_end_times))
    # exit()
    # Set checkpoints
    snapshots = []
    snapshots.append({
        'snapshot index': 0, # This index is the first operation after this checkpoint
        'snapshot vars': all_input_vars, # This is the set of all variables in this snapshot
        'snapshot memory': input_vars_cost, # Total amount of memory needed for getting to this snapshot from the previous
        'snapshot operations': set(), # Operations to be executed from last snapshot to this snapshot
        'snapshot schedule': [[] for _ in range(number_of_ranks)],  # Same as snapshot operation but ordered
        'idle time': 0.0, # Amount of idle time between this snapshot and the previous. The lower, the better the load balancing
    })
    best_potential_idle_time = np.inf
    best_end_time_index = np.inf
    i = 0
    while i < len(ordered_end_times):
        end_time = ordered_end_times[i]
        current_memory = mapped_end_times[end_time]['maximum prefix memory allocation']
        
        # See if we add snapshot
        previous_snapshot_end_time = ordered_end_times[snapshots[-1]['snapshot index']]
        previous_snapshot_max_memory = mapped_end_times[previous_snapshot_end_time]['maximum prefix memory allocation']
        memory_this_checkpoint = current_memory - previous_snapshot_max_memory
        # print(i, memory_this_checkpoint, current_memory, len(snapshots), previous_snapshot_max_memory)

        # Everything point after this is potentially a checkpoint. Keep track of minimum idle time.
        if (memory_this_checkpoint > min_stride) or (i == (len(mapped_end_times)-1)):
            
            # Compute idle time for checkpoint
            idle_time = 0
            for rank in range(number_of_ranks):
                idle_time += abs(end_time - mapped_end_times[end_time]['latest tasks'][rank][2])

            # Check if we are better than the best potential idle time
            if idle_time <= best_potential_idle_time:

                num_sends = 0
                num_recvs = 0
                latest_tasks = mapped_end_times[end_time]['latest tasks']
                for op in latest_tasks:
                    if 'SEND_/' in op:
                        num_sends += 1
                    elif 'GET_/' in op:
                        num_recvs += 1
                if num_sends == num_recvs:
                    best_potential_idle_time = idle_time
                    best_end_time_index = i

            # Here, we actually set the checkpoint based on the best idle time.
            if (memory_this_checkpoint > max_stride) or (i == (len(mapped_end_times)-1)):
                if i == (len(mapped_end_times)-1):
                    best_end_time_index = i + 1
                # Previous checkpoint
                prev_checkpoint = snapshots[-1]
                previous_checkpoint_end_time_index = prev_checkpoint['snapshot index']

                # print('checkpoint')
                # Get snapshot variables:
                outputs_to_this_checkpoint = set()
                operations_in_this_checkpoint = set()
                operation_schedule = [[] for _ in range(number_of_ranks)]
                for j in range(previous_checkpoint_end_time_index, best_end_time_index):
                    for rank in range(number_of_ranks):
                        current_operation = mapped_end_times[ordered_end_times[j]]['latest tasks'][rank][0]

                        if current_operation is None:
                            continue
                        
                        add_op = True
                        for previous_snaps in snapshots:
                            if current_operation in previous_snaps['snapshot operations']:
                                add_op = False
                        if not add_op:
                            continue
                        # previous_snaps = snapshots[-1]
                        # if current_operation in previous_snaps['snapshot operations']:
                        #     add_op = False
                        #     continue
                        if current_operation not in operations_in_this_checkpoint:
                            operation_schedule[rank].append(current_operation)
                        operations_in_this_checkpoint.add(current_operation)

                        if current_operation not in graph.nodes:
                            continue

                        for successor in graph.successors(current_operation):
                            outputs_to_this_checkpoint.add(successor)

                # Calculate variables we need to actually keep:
                for output in outputs_to_this_checkpoint.copy():
                    delete_output = True
                    for successor in graph.successors(output):
                        # successor is an operation
                        if successor not in operations_in_this_checkpoint:
                            delete_output = False
                    if graph.out_degree(output) == 0:
                        delete_output = False

                    if delete_output:
                        outputs_to_this_checkpoint.remove(output)

                if (i == (len(mapped_end_times)-1)):
                    snap_shot_memory = mapped_end_times[ordered_end_times[best_end_time_index-1]]['maximum prefix memory allocation'] - mapped_end_times[ordered_end_times[snapshots[-1]['snapshot index']]]['maximum prefix memory allocation']
                else:
                    snap_shot_memory = mapped_end_times[ordered_end_times[best_end_time_index]]['maximum prefix memory allocation'] - mapped_end_times[ordered_end_times[snapshots[-1]['snapshot index']]]['maximum prefix memory allocation']
                # Set new checkpoint
                # import random
                # el = random.sample(outputs_to_this_checkpoint, 1)[0]
                # outputs_to_this_checkpoint.remove(el)
                snapshots.append({
                    'snapshot index': best_end_time_index,
                    'snapshot vars': outputs_to_this_checkpoint,
                    'snapshot memory': snap_shot_memory,
                    'snapshot operations': operations_in_this_checkpoint,
                    'snapshot schedule': operation_schedule,
                    'idle time': best_potential_idle_time,
                })
                # print('CHECKPOINT',operations_in_this_checkpoint)

                # reset variables
                i = best_end_time_index
                best_potential_idle_time = np.inf
                best_end_time_index = np.inf
        
        i = i + 1
    # Check:
    for end_time in mapped_end_times:
        # print(end_time, mapped_end_times[end_time]['maximum prefix memory allocation'], ' Bytes\t', [round(last_tasks[-1], 5) for last_tasks in mapped_end_times[end_time]['latest tasks']])
        for rank in range(number_of_ranks):
            if mapped_end_times[end_time]['latest tasks'][rank][2] is None:
                continue
            if mapped_end_times[end_time]['latest tasks'][rank][2] > end_time+1e-6:
                # raise ValueError('esldfnsf')
                raise ValueError(f'end time not less than end time ({ mapped_end_times[end_time][rank][2]} !<= {end_time})')
    
    # Check to make sure snapshots are correct
    # 1) Union of (all operations in a checkpoint snapshot) for all snapshots should equal all operations
    # 2) Intesection of (all operations in a checkpoint snapshot) for all snapshots should be empty
    # 3) The predecessors of all snapshot variables should have be included in a previous checkpoint
    # 4) numbers of sends and receives in a snapshot interval should be equal to prevent deadlock
    # 5) The predecessors of all snapshot operations not computed in the snapshot should have been included in a previous snapshot
    # 6) operation schedule should have exact same operations as snapshot operations and must be in same order as uncheckpointed schedule
    previous_operations = set()
    previous_snapshot_variables = set()
    num_total_scheduled = len(all_scheduled_ops)
    global_schedule_from_checkpoints = [[] for _ in range(number_of_ranks)] 
    for i, snapshot in enumerate(snapshots):
        num_ops_this_snapshot = len(snapshot['snapshot operations'])
        print('Checking snapshot', i)
        print('\tindex:           ', snapshot['snapshot index'])
        print('\t# save vars:     ', len(snapshot['snapshot vars']))
        print('\t# operations:    ', f'({num_ops_this_snapshot+len(previous_operations)}/{num_total_scheduled})')
        print('\tinterval memory: ', snapshot['snapshot memory'])
        print('\tidle time:       ', snapshot['idle time'])
        num_sends = 0
        num_recvs = 0

        current_snapshot_subgraph_operations = set()
        for operation in snapshot['snapshot operations']:
            # print(operation)
            if 'SEND_/' in operation:
                num_sends += 1
            elif 'GET_/' in operation:
                num_recvs += 1

            previous_operations.add(operation)

            # Check 2)
            if operation not in all_scheduled_ops:
                raise ValueError(f'Operation was already scheduled in a previous checkpoint ({operation})')
            all_scheduled_ops.remove(operation)

            if operation in graph.nodes:
                current_snapshot_subgraph_operations.add(operation)

        # Check 5)
        snapshot_subgraph = graph.subgraph(current_snapshot_subgraph_operations)
        for operation in snapshot_subgraph.nodes:
            for predecessor in graph.predecessors(operation):
                for predecessor_predecessor in graph.predecessors(predecessor):
                    if predecessor_predecessor not in snapshot['snapshot operations']:
                        if predecessor not in previous_snapshot_variables:
                            raise ValueError(f'Predecessor of snapshot operation was not included in a previous snapshot (variable {predecessor})')
        
        # Check 4)
        if num_sends != num_recvs:
            print(total_sends, total_recvs)
            raise ValueError(f'Number of sends and receives are not equal in a checkpoint. Potential deadlock ({num_sends} send != {num_recvs} recvs)')

        # Check 3)
        for snapshot_var in snapshot['snapshot vars']:
            previous_snapshot_variables.add(snapshot_var)
            for predecessor in graph.predecessors(snapshot_var):
                if predecessor not in previous_operations:
                    raise ValueError(f'Predecessor of snapshot variable was not included in a previous checkpoint (variable {predecessor})')

        # Check 6) a)
        all_ops_from_schedule = set().union(*[set(schedule) for schedule in snapshot['snapshot schedule']])
        if all_ops_from_schedule != snapshot['snapshot operations']:
            raise ValueError(f'Operation schedule does not match snapshot operations (snapshot {i})')
        for rank in range(number_of_ranks):
            global_schedule_from_checkpoints[rank].extend(snapshot['snapshot schedule'][rank])
    # Check 1)
    if len(all_scheduled_ops) != 0:
        # for snapshot in snapshots:
        #     print('SNAPSHOT', snapshot['snapshot index'])
        #     for operation in snapshot['snapshot operations']:
        #         print(operation)
        # print(all_scheduled_ops)
        raise ValueError('Not all operations were scheduled in a checkpoint')
    
    # for rank in range(number_of_ranks):
        # print(rank, global_schedule_from_checkpoints[rank])
        # print(rank,schedules[rank])
    # Check 6) b)
    for rank in range(number_of_ranks):
        i = 0
        for op in schedules[rank]:
            if 'WAIT' in op:
                continue
            if op != global_schedule_from_checkpoints[rank][i]:
                raise ValueError(f'Operation schedule does not match order of snapshot operation schedule (rank {rank}, {op} vs {global_schedule_from_checkpoints[rank][i]}, index {i})')
            i+=1
    

    
    # print snapshots:
    # for i, snapshot in enumerate(snapshots):
    #     print('SNAPSHOT', snapshot['snapshot index'])
    #     print('\tindex:         ', i)
    #     print('\t# save vars:    ', len(snapshot['snapshot vars']))
    #     print('\tmemory used    ', snapshot['snapshot memory'])
    #     print('\tidle time    ', snapshot['idle time'])

    # Now we have to split all snapshots into their own ranks
    
    return snapshots
    
            