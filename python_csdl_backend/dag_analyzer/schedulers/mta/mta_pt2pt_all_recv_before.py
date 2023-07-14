from python_csdl_backend.dag_analyzer.schedulers.algorithm import Algorithm
from python_csdl_backend.dag_analyzer.graph_generator.sdag_to_op_only_graph import sdag_to_op_only_graph

from python_csdl_backend.dag_analyzer.schedulers.mta.compute_priorities import compute_furthest_weight_ahead
from python_csdl_backend.dag_analyzer.schedulers.scheduler_functions import schedule_pt2pt_comm, schedule_operation, get_pt2pts_key, add_to_schedule, add_to_estimated_timeline

from python_csdl_backend.dag_analyzer.utils import draw

class MTA_PT2PT_ARB(Algorithm):
    
    def schedule(self, ccl_graph):

        NUM_RANKS = self.NUM_RANKS
        sdag = ccl_graph
        if self.create_plots:
            draw(sdag, title = 'images/SDAG')

        # Translate standard graph to operation-only graph
        o_sdag, total_weight = sdag_to_op_only_graph(sdag)
        if self.create_plots:
            draw(o_sdag, title = 'images/OSDAG')

        compute_furthest_weight_ahead(o_sdag)

        # Schedule data structure:
        # schedule = [[], [], [], ... (NUM_RANKS times)]
        preliminary_schedule = []
        preliminary_estimated_timeline = []
        for _ in range(NUM_RANKS):
            preliminary_schedule.append([])
            preliminary_estimated_timeline.append([0])

        # Preprocessing
        import heapq
        available_operations_heap = [] # A is a heap containing available nodes where heap root is the best to schedule
        input_operations = set()
        output_variables = set()
        input_variables = set()
        # Find all available nodes
        for node in sdag.nodes:
            # longest_time_ahead = o_sdag.nodes[node]['FWA']
            # print(longest_time_ahead, node)
            if sdag.nodes[node]['type'] == 'operation':
                o_sdag.nodes[node]['touches_left'] = o_sdag.in_degree(node)
                o_sdag.nodes[node]['touches_left_final'] = o_sdag.in_degree(node)
                o_sdag.nodes[node]['release_time'] = 'NOT_READY'
                o_sdag.nodes[node]['release_time_final'] = 'NOT_READY'

                if o_sdag.in_degree(node) == 0:
                    if node not in input_operations:
                        input_operations.add(node)
                        longest_time_ahead = o_sdag.nodes[node]['FWA']
                        heapq.heappush(available_operations_heap, (longest_time_ahead, node))
            
            if sdag.in_degree(node) == 0:
                # If a input leaf operation has a non-zero rank, we need to send the inputs from rank zero to the appropriate rank
                input_variables.add(node)

            if sdag.out_degree(node) == 0:
                output_variables.add(node)
                # If a output leaf operation has a non-zero rank, we need to send the outputs from the appropriate rank to rank zero
                sdag.nodes[node]['output'] = 1

        # Main loop:
        while available_operations_heap:
            v_information = heapq.heappop(available_operations_heap) # Get best node to schedule
            v  = v_information[1] # v is the current node to schedule
            # print(v)

            # Algorithm:
            # **Estimate** the earliest possible start time for node v for each rank v_rank
            # Loop through each rank, determine the rank that would start this operation the fastest
            potential_v_schedules = []
            COMM_COST = 0
            for v_rank in range(NUM_RANKS):
                # if we want to schedule v on v_rank, it must start after most recent job on v_rank is finished or after all its predecessors are finished
                minimum_v_start_time = preliminary_estimated_timeline[v_rank][-1]
                for p in o_sdag.predecessors(v):
                    # p is an operation node
                    p_rank = o_sdag.nodes[p]['rank']
                    if p_rank != v_rank:
                        # Find the minimum time needed to schedule v on v_rank
                        for p_var in o_sdag.edges[(p,v)]['edge_variables']:
                            # s_var are variables computed from p that feeds into v
                            COMM_COST = ccl_graph.nodes[p_var]['cost']
                        
                        minimum_v_start_time = max(o_sdag.nodes[p]['release_time'] + COMM_COST, minimum_v_start_time + COMM_COST)

                if v_rank != 0:
                    for p in sdag.predecessors(v):
                        if sdag.in_degree(p) == 0:
                            COMM_COST = ccl_graph.nodes[p]['cost']
                            minimum_v_start_time = max(COMM_COST, minimum_v_start_time)
                # print('\t',v_rank, minimum_v_start_time)
                # compute start time of v and created idle
                v_start_time = minimum_v_start_time
                idle_time = minimum_v_start_time - preliminary_estimated_timeline[v_rank][-1]
                potential_v_schedules.append((v_start_time, idle_time, v_rank))

            # for each rank, sort by start time of v
            potential_v_schedules = sorted(potential_v_schedules)
            DELTA = 0.0
            minimum_start = potential_v_schedules[0][0]
            minimum_idle = potential_v_schedules[0][1]
            best_rank = potential_v_schedules[0][2]
            for v_schedule_info in potential_v_schedules:
                if v_schedule_info[0] > (potential_v_schedules[0][0] + DELTA):
                    break
                # at this point, this rank is a candidate for scheduling v
                if v_schedule_info[1] < minimum_idle:
                    minimum_start = v_schedule_info[0]
                    minimum_idle = v_schedule_info[1]
                    best_rank = v_schedule_info[2]
            v_rank = best_rank

            # CREATE PRELIMINARY SCHEDULE FOR OPERATIONS
            OP_COST = ccl_graph.nodes[v]['cost']
            o_sdag.nodes[v]['rank'] = v_rank

            if (minimum_start - preliminary_estimated_timeline[v_rank][-1]) != 0:
                schedule_operation(
                    preliminary_schedule,
                    schedule_estimated_timeline = preliminary_estimated_timeline,
                    operation = f'W/F{v}',
                    operation_cost=minimum_start - preliminary_estimated_timeline[v_rank][-1],
                    rank = v_rank,
                )
            schedule_operation(
                preliminary_schedule,
                schedule_estimated_timeline = preliminary_estimated_timeline,
                operation = v,
                operation_cost=OP_COST,
                rank = v_rank,
            )
            o_sdag.nodes[v]['release_time'] = preliminary_estimated_timeline[v_rank][-1]
            # Update heap
            for s in o_sdag.successors(v):
                # If all input variables to s have been computed, s is available to compute
                o_sdag.nodes[s]['touches_left'] = o_sdag.nodes[s]['touches_left'] - 1
                if o_sdag.nodes[s]['touches_left'] == 0:
                    longest_time_ahead = o_sdag.nodes[s]['FWA']
                    heapq.heappush(available_operations_heap, (longest_time_ahead, s))
        
        for node in o_sdag.nodes:
            if o_sdag.nodes[node]['touches_left'] != 0:
                raise ValueError('ERROR')
        # =-=-==-==-=-=-=-=-=-=- FINAL SCHEDULING =-=-==-=-=-=-=-=-=-=-=-=--==-
        # Now that we know the operation schedule for each rank,
        # build the full schedule with pt2pt communication.

        # How do we do this ...
        # We know which rank each operation is scheduled on
        # The only think we have to do is actually schedule the recieves and sends for each rank
        # We want to schedule sends as soon as we can.
        # - Whenever we process an operation, lets send it to all successor ranks immediately
        # We also need to schedule recieves (irecieves to be exact) at the BEGINNING of each rank schedule 
        # - Whenever we issue a send (above^), add an irecieve to a prefix schedule
        # at the end, final schedule = prefix_schedule + schedule
        # What do we do about the estimated timeline?

        # initialize data structures
        schedule = [] # schedule operations and mpi sends and mpi receives
        estimated_timeline = [] # approximation of start/end time for each item in schedule
        preliminary_schedule_operation = []
        wait_recv_maps = []
        sent_from_already = []
        for _ in range(NUM_RANKS):
            schedule.append([])
            sent_from_already.append(set())
            wait_recv_maps.append({})
            preliminary_schedule_operation.append([])
            estimated_timeline.append([0])

        rank_zero_waits = set()

        # for i, tl in enumerate(preliminary_estimated_timeline):
        #     print(i, preliminary_schedule[i])
        #     print(i, tl)


        # Schedule all irecieves before scheduling operations
        # We just loop through all ranks and their scheduled operations
        # Then we just schedule whatever we need
        for v_rank in range(NUM_RANKS):
            received_vars = set()
            
            for v in preliminary_schedule[v_rank]:
                if 'W/F' in v:
                    continue
                preliminary_schedule_operation[v_rank].append(v)
                for p in o_sdag.predecessors(v):
                    p_rank = o_sdag.nodes[p]['rank']
                    if p_rank != v_rank:
                        for p_var in o_sdag.edges[(p,v)]['edge_variables']:
                            if p_var not in received_vars:
                                received_vars.add(p_var)
                                schedule_pre_irecieve(
                                    full_schedule = schedule,
                                    schedule_estimated_timeline= estimated_timeline,
                                    variable = p_var,
                                    recieve_cost= 0, # sdag.nodes[p_var]['cost']
                                    to_rank = v_rank,
                                    from_rank = p_rank,
                                )
                                if v not in wait_recv_maps[v_rank]:
                                    wait_recv_maps[v_rank][v] = {p_var}
                                else:
                                    wait_recv_maps[v_rank][v].add(p_var)


                if v_rank != 0:
                    for p_var in sdag.predecessors(v):
                        if sdag.in_degree(p_var) == 0:
                            if p_var not in received_vars:
                                received_vars.add(p_var)
                                schedule_pre_irecieve(
                                    full_schedule = schedule,
                                    schedule_estimated_timeline= estimated_timeline,
                                    variable = p_var,
                                    recieve_cost= 0, # sdag.nodes[p_var]['cost']
                                    to_rank = v_rank,
                                    from_rank = 0,
                                )
                                if v not in wait_recv_maps[v_rank]:
                                    wait_recv_maps[v_rank][v] = {p_var}
                                else:
                                    wait_recv_maps[v_rank][v].add(p_var)

                                schedule_send_only( 
                                    full_schedule = schedule,
                                    schedule_estimated_timeline= estimated_timeline,
                                    variable = p_var,
                                    send_cost= sdag.nodes[p_var]['cost'],
                                    to_rank = v_rank,
                                    from_rank = 0,
                                )
        # for i in estimated_timeline:
        #     print(i[-1])
        # exit()

        occupied_ranks = set()
        reversed_preliminary_schedule = []
        for rank in range(len(preliminary_schedule_operation)):
            reversed_preliminary_schedule.append(list(reversed(preliminary_schedule_operation[rank])))
            occupied_ranks.add(rank)

        # Schedule operations
        while occupied_ranks:
            v_rank = None
            # Find least burdened rank that still needs to be processed
            remove_ranks = set()
            for i, rank in enumerate(occupied_ranks):
                if len(reversed_preliminary_schedule[rank]) == 0:
                    remove_ranks.add(rank)
                    continue

                # make sure that the next operation in rank is available to schedule
                candidate_op = reversed_preliminary_schedule[rank][-1]
                tlf = o_sdag.nodes[candidate_op]['touches_left_final']
                # print(candidate_op, tlf)
                if tlf > 0:
                    continue
                
                # find best operation to schedule
                rank_current_time = estimated_timeline[rank][-1]
                if v_rank is None:
                    v_rank = rank
                    min_time = rank_current_time
                elif rank_current_time < min_time:
                    min_time = rank_current_time
                    v_rank = rank
            # If the rank has all operations scheduled, it is no longer occupied and we exit the loop.
            for rank in remove_ranks:
                occupied_ranks.remove(rank)
                
            if len(occupied_ranks) == 0:
                continue
            
            # print(remove_ranks, occupied_ranks, len(reversed_preliminary_schedule[0]))
            v = reversed_preliminary_schedule[v_rank].pop() # v is operation to schedule
            v_rank = o_sdag.nodes[v]['rank']
            
            op_start_time = min_time
            for p in o_sdag.predecessors(v):
                p_rank = o_sdag.nodes[p]['rank']
                if p_rank != v_rank:
                    for p_var in o_sdag.edges[(p,v)]['edge_variables']:
                        # op_start_time = max(o_sdag.nodes[p]['release_time_final']+COMM_COST, op_start_time)
                        op_start_time = max(o_sdag.nodes[p]['release_time_final'], op_start_time)

            if v_rank != 0:
                for p_var in sdag.predecessors(v):
                    if sdag.in_degree(p_var) == 0:
                        # op_start_time = op_start_time+COMM_COST
                        op_start_time = op_start_time


            if (op_start_time - estimated_timeline[v_rank][-1]) != 0:
                schedule_operation(
                    schedule,
                    schedule_estimated_timeline = estimated_timeline,
                    operation = f'W/F{v}',
                    operation_cost=op_start_time - estimated_timeline[v_rank][-1],
                    rank = v_rank,
                )

            if v in wait_recv_maps[v_rank]:
                for wait_var in wait_recv_maps[v_rank][v]:
                    schedule_irecv_wait(
                        schedule,
                        estimated_timeline,
                        wait_var,
                        v_rank,
                    )

            OP_COST = ccl_graph.nodes[v]['cost']
            schedule_operation(
                schedule,
                schedule_estimated_timeline = estimated_timeline,
                operation = v,
                operation_cost=OP_COST,
                rank = v_rank,
            )
            # o_sdag.nodes[v]['release_time_final'] = preliminary_estimated_timeline[v_rank][-1]
            o_sdag.nodes[v]['release_time_final'] = estimated_timeline[v_rank][-1]


            sent_to_already = []
            for _ in range(NUM_RANKS):
                sent_to_already.append(set())
            for s in o_sdag.successors(v):

                # If all input variables to s have been computed, s is available to compute
                o_sdag.nodes[s]['touches_left_final'] = o_sdag.nodes[s]['touches_left_final'] - 1

                s_rank = o_sdag.nodes[s]['rank']
                if s_rank != v_rank:
                    for s_var in o_sdag.edges[(v,s)]['edge_variables']:

                        if s_var not in sent_to_already[s_rank]:
                            sent_to_already[s_rank].add(s_var)
                            schedule_send_only( 
                                full_schedule = schedule,
                                schedule_estimated_timeline= estimated_timeline,
                                variable = s_var,
                                send_cost= sdag.nodes[s_var]['cost'],
                                to_rank = s_rank,
                                from_rank = v_rank,
                            )


            if v_rank != 0:
                for successor in sdag.successors(v):
                    if 'output' in sdag.nodes[successor]:
                        schedule_pre_irecieve(
                            full_schedule = schedule,
                            schedule_estimated_timeline= estimated_timeline,
                            variable = successor,
                            recieve_cost= 0, # sdag.nodes[p_var]['cost']
                            to_rank = 0,
                            from_rank = v_rank,
                        )

                        schedule_send_only( 
                            full_schedule = schedule,
                            schedule_estimated_timeline= estimated_timeline,
                            variable = successor,
                            send_cost= sdag.nodes[p_var]['cost'],
                            to_rank = 0,
                            from_rank = v_rank,
                        )

                        rank_zero_waits.add(successor)


        for output in rank_zero_waits:
            schedule_irecv_wait(
                schedule,
                estimated_timeline,
                output,
                0,
            )

        # for i, tl in enumerate(estimated_timeline):
        #     print(i, schedule[i])
        #     print(i, tl)
        # exit('slkdfns')

        return schedule, estimated_timeline


def schedule_pre_irecieve(
        full_schedule: list[list[str]],
        schedule_estimated_timeline: list[list[int]],
        variable: str,
        recieve_cost: float,
        to_rank: int,
        from_rank:int,
    ):

    add_to_schedule(full_schedule, f'IRECV_/{variable}/_from_/{from_rank}/', to_rank)
    add_to_estimated_timeline(schedule_estimated_timeline, schedule_estimated_timeline[to_rank][-1]+recieve_cost, to_rank)

def schedule_send_only(
        full_schedule: list[list[str]],
        schedule_estimated_timeline: list[list[int]],
        variable: str,
        send_cost: float,
        to_rank: int,
        from_rank:int,
    ):

    add_to_schedule(full_schedule, f'SsENDONLY_/{variable}/_to_/{to_rank}/', from_rank)
    add_to_estimated_timeline(schedule_estimated_timeline, schedule_estimated_timeline[from_rank][-1]+send_cost, from_rank)


def schedule_irecv_wait(
        full_schedule: list[list[str]],
        schedule_estimated_timeline: list[list[int]],
        variable: str,
        rank = int,
    ):

    add_to_schedule(full_schedule, f'irecvwait_/{variable}', rank)
    add_to_estimated_timeline(schedule_estimated_timeline, schedule_estimated_timeline[rank][-1], rank)