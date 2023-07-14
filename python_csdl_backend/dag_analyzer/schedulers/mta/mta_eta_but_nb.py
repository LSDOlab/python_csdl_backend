from python_csdl_backend.dag_analyzer.schedulers.algorithm import Algorithm
from python_csdl_backend.dag_analyzer.graph_generator.sdag_to_op_only_graph import sdag_to_op_only_graph

from python_csdl_backend.dag_analyzer.schedulers.mta.compute_priorities import compute_furthest_weight_ahead
from python_csdl_backend.dag_analyzer.schedulers.scheduler_functions import schedule_pt2pt_comm, schedule_operation

from python_csdl_backend.dag_analyzer.utils import draw

class MTA_PT2PT_ETA_BUT_NB(Algorithm):
    
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
        # schedule = [[], [], [], ... (NUMBER_OF_PARTS times)]
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

                if o_sdag.in_degree(node) == 0:
                    if node not in input_operations:
                        input_operations.add(node)
                        longest_time_ahead = o_sdag.nodes[node]['FWA']
                        heapq.heappush(available_operations_heap, (longest_time_ahead, node))
            
            if sdag.in_degree(node) == 0:
                # If a input leaf operation has a non-zero rank, we need to send the inputs from rank zero to the appropriate rank
                input_variables.add(node)

                # Also, initialize the available operations heap
                # op_nodes = sdag.successors(node)
                # for op in op_nodes:
                #     if op not in input_operations:
                #         input_operations.add(op)
                #         longest_time_ahead = o_sdag.nodes[op]['FWA']
                #         heapq.heappush(available_operations_heap, (longest_time_ahead, op))

            if sdag.out_degree(node) == 0:
                output_variables.add(node)
                # If a output leaf operation has a non-zero rank, we need to send the outputs from the appropriate rank to rank zero
                sdag.nodes[node]['output'] = 1

        
        # exit()
        # Main loop:
        global_operation_ordering = []
        while available_operations_heap:
            v_information = heapq.heappop(available_operations_heap) # Get best node to schedule
            v  = v_information[1] # v is the current node to schedule
            # ================== NEW ==================
            # Algorithm:
            # **Estimate** the earliest possible start time for node v for each rank v_rank
            potential_v_schedules = []
            COMM_COST = 0
            for v_rank in range(NUM_RANKS):
                # if we want to schedule v on v_rank, it must start after most recent job on v_rank is finished.
                minimum_v_start_time = preliminary_estimated_timeline[v_rank][-1]
                for p in o_sdag.predecessors(v):
                    # p is an operation node
                    p_rank = o_sdag.nodes[p]['rank']
                    if p_rank != v_rank:
                        # Find the minimum time needed to schedule v on v_rank
                        for p_var in o_sdag.edges[(p,v)]['edge_variables']:
                            # s_var are variables computed from p that feeds into v
                            COMM_COST = ccl_graph.nodes[p_var]['cost']
                        
                        minimum_v_start_time = max(o_sdag.nodes[p]['release_time'] + COMM_COST, minimum_v_start_time)
                            
                # compute start time of v and created idle
                v_start_time = minimum_v_start_time
                idle_time = minimum_v_start_time - preliminary_estimated_timeline[v_rank][-1]
                potential_v_schedules.append((v_start_time, idle_time, v_rank))

            # for each rank, sort by start time of v
            potential_v_schedules = sorted(potential_v_schedules)
            DELTA = 1.5*COMM_COST
            minimum_idle = potential_v_schedules[0][1]
            best_rank = potential_v_schedules[0][2]
            for v_schedule_info in potential_v_schedules:
                if v_schedule_info[0] > (potential_v_schedules[0][0] + DELTA):
                    break
                # at this point, this rank is a candidate for scheduling v
                if v_schedule_info[1] < minimum_idle:
                    minimum_idle = v_schedule_info[1]
                    best_rank = v_schedule_info[2]
            v_rank = best_rank
            # ================== NEW ==================

            OP_COST = ccl_graph.nodes[v]['cost']
            o_sdag.nodes[v]['rank'] = v_rank
            schedule_operation(
                preliminary_schedule,
                schedule_estimated_timeline = preliminary_estimated_timeline,
                operation = v,
                operation_cost=OP_COST,
                rank = v_rank
            )            # v_rank = o_sdag.nodes[v]['rank']
            o_sdag.nodes[v]['release_time'] = preliminary_estimated_timeline[v_rank][-1]
            global_operation_ordering.append(v)
            # Update heap
            for s in o_sdag.successors(v):
                # If all input variables to s have been computed, s is available to compute
                o_sdag.nodes[s]['touches_left'] = o_sdag.nodes[s]['touches_left'] - 1
                if o_sdag.nodes[s]['touches_left'] == 0:
                    longest_time_ahead = o_sdag.nodes[s]['FWA']
                    heapq.heappush(available_operations_heap, (longest_time_ahead, s))

        # =-=-==-==-=-=-=-=-=-=- FINAL SCHEDULING =-=-==-=-=-=-=-=-=-=-=-=--==-
        # Now that we know the operation schedule for each rank,
        # build the full schedule with pt2pt communication.
        
        # initialize data structures
        schedule = []
        estimated_timeline = []
        for _ in range(NUM_RANKS):
            schedule.append([])                
            estimated_timeline.append([0])
        
        occupied_ranks = set()
        reversed_preliminary_schedule = []
        for rank in range(len(preliminary_schedule)):
            reversed_preliminary_schedule.append(list(reversed(preliminary_schedule[rank])))
            occupied_ranks.add(rank)
            # print('rank:',rank , preliminary_schedule[rank])
        
        # preprocessing:
        for node in input_variables:
            op_nodes = sdag.successors(node)
            for op in op_nodes:

                # If rank of input operation is NOT 0, send the input variables from rank 0 to correct rank
                op_rank = o_sdag.nodes[op]['rank']
                if op_rank != 0:
                    COMM_COST = ccl_graph.nodes[node]['cost']
                    schedule_pt2pt_comm(
                        schedule,
                        schedule_estimated_timeline = estimated_timeline,
                        node = node,
                        communication_cost= COMM_COST,
                        from_rank = 0,
                        to_rank = op_rank)

        # main scheduleing loop
        # while occupied_ranks:
        #     v_rank = -1
        #     # Find least burdened rank that still needs to be processed
        #     for i, rank in enumerate(occupied_ranks):

        #         # make sure that the next operation in rank is available to schedule
        #         if len(reversed_preliminary_schedule[rank]) > 0:
        #             candidate_op = reversed_preliminary_schedule[rank][-1]
        #             tlf = o_sdag.nodes[candidate_op]['touches_left_final']
        #             if tlf > 0:
        #                 continue
                
        #         # find best operation to schedule
        #         rank_current_time = estimated_timeline[rank][-1]
        #         if v_rank == -1:
        #             v_rank = rank
        #             min_time = rank_current_time
        #         elif rank_current_time < min_time:
        #             min_time = rank_current_time
        #             v_rank = rank
        #     # If the rank has all operations scheduled, it is no longer occupied.
        #     if len(reversed_preliminary_schedule[v_rank]) == 0:
        #         occupied_ranks.remove(v_rank)
        #         continue
        #     Let's schedule the next available operation (called v) on rank v_rank!                
        #     v = reversed_preliminary_schedule[v_rank].pop() # v is operation to schedule
        for v in global_operation_ordering:
            # Let's schedule the next available operation (called v) on rank v_rank!                
            # v = reversed_preliminary_schedule[v_rank].pop() # v is operation to schedule
            v_rank = o_sdag.nodes[v]['rank']
            # print(v_rank, v, reversed_preliminary_schedule[v_rank])
            # First schedule the operation
            OP_COST = ccl_graph.nodes[v]['cost']
            schedule_operation(
                schedule,
                schedule_estimated_timeline = estimated_timeline,
                operation = v,
                operation_cost=OP_COST,
                rank = v_rank
            )

            # Now let's schedule all pt2pt communications to send outputs of v
            # to any operations on other ranks that need it!!!!!!!!!!!!!!!!!
            for s in o_sdag.successors(v):
                o_sdag.nodes[s]['touches_left_final'] = o_sdag.nodes[s]['touches_left_final'] - 1
                # s is an operation node
                s_rank = o_sdag.nodes[s]['rank']
                if s_rank != v_rank:
                    for s_var in o_sdag.edges[(v,s)]['edge_variables']:
                        # s_var are variables computed from v that feeds into s
                        schedule_pt2pt_comm(
                            schedule,
                            schedule_estimated_timeline = estimated_timeline,
                            node = s_var,
                            communication_cost= COMM_COST,
                            from_rank=v_rank,
                            to_rank=s_rank,
                        )

        # post-processing
        for node in output_variables:
            preds = list(sdag.predecessors(node))
            if len(preds) > 0:
                op = preds[0]
                op_rank = o_sdag.nodes[op]['rank']
                if op_rank != 0:
                    COMM_COST = ccl_graph.nodes[node]['cost']
                    schedule_pt2pt_comm(
                        schedule,
                        schedule_estimated_timeline = estimated_timeline,
                        node = node,
                        communication_cost= COMM_COST,
                        from_rank=op_rank,
                        to_rank=0)

        return schedule, estimated_timeline
