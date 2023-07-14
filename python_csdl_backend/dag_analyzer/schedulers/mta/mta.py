from python_csdl_backend.dag_analyzer.schedulers.algorithm import Algorithm
from python_csdl_backend.dag_analyzer.graph_generator.sdag_to_op_only_graph import sdag_to_op_only_graph

from python_csdl_backend.dag_analyzer.schedulers.mta.compute_priorities import compute_furthest_weight_ahead
from python_csdl_backend.dag_analyzer.schedulers.scheduler_functions import schedule_pt2pt_comm, schedule_operation

from python_csdl_backend.dag_analyzer.utils import draw

class MTA(Algorithm):
    
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
        schedule = []
        estimated_timeline = []
        for rank in range(NUM_RANKS):
            schedule.append([])                
            estimated_timeline.append([0])

        # Preprocessing
        import heapq
        available_operations_heap = [] # A is a heap containing available nodes where heap root is the best to schedule
        input_operations = set()
        output_variables = set()
        # Find all available nodes
        for node in sdag.nodes:
            # longest_time_ahead = o_sdag.nodes[node]['FWA']
            # print(longest_time_ahead, node)
            if sdag.nodes[node]['type'] == 'operation':
                o_sdag.nodes[node]['touches_left'] = o_sdag.in_degree(node)
            if sdag.in_degree(node) == 0:
                
                # If a input leaf operation has a non-zero rank, we need to send the inputs from rank zero to the appropriate rank
                # Also, initialize the available operations heap
                op_nodes = sdag.successors(node)
                for op in op_nodes:
                    if op not in input_operations:
                        input_operations.add(op)
                        longest_time_ahead = o_sdag.nodes[op]['FWA']
                        heapq.heappush(available_operations_heap, (longest_time_ahead, op))

                    # If rank of input operation is NOT 0, send the input variables from rank 0 to correct rank
                    for op_rank in range(NUM_RANKS):
                        if op_rank != 0:
                            COMM_COST = ccl_graph.nodes[node]['cost']
                            schedule_pt2pt_comm(
                                schedule,
                                schedule_estimated_timeline = estimated_timeline,
                                node = node,
                                communication_cost= COMM_COST,
                                from_rank = 0,
                                to_rank = op_rank)

            if sdag.out_degree(node) == 0:
                output_variables.add(node)
                # If a output leaf operation has a non-zero rank, we need to send the outputs from the appropriate rank to rank zero
                sdag.nodes[node]['output'] = 1

        
        # exit()
        # Main loop:
        while available_operations_heap:
            v_information = heapq.heappop(available_operations_heap) # Get best node to schedule
            v  = v_information[1] # v is the current node to schedule

            # ================== OLD ==================
            # Find rank with least workload so far.
            # This is where we schedule the operation.
            # for rank in range(NUM_RANKS):
            #     if rank == 0:
            #         min_workload = estimated_timeline[rank][-1]
            #         least_burdened_rank = rank
            #     else:
            #         if estimated_timeline[rank][-1] < min_workload:
            #             min_workload = estimated_timeline[rank][-1]
            #             least_burdened_rank = rank
            # v_rank = least_burdened_rank # v_rank is the rank that v is supposed to be schedule on
            # ================== OLD ==================
            
            # ================== NEW ==================
            # Algorithm:
            # Simulate scheduling v on all ranks and keep track of ranks and their time to schedule v
            potential_v_schedules = []
            for v_rank in range(NUM_RANKS):
                
                dummy_schedule = []
                dummy_estimated_timeline = []
                for rank in range(NUM_RANKS):

                    if len(schedule[rank]) == 0:
                        dummy_schedule.append([0])
                    else:
                        dummy_schedule.append([schedule[rank][-1]])
                    dummy_estimated_timeline.append([estimated_timeline[rank][-1]])

                for p in o_sdag.predecessors(v):
                    # p is an operation node
                    p_rank = o_sdag.nodes[p]['rank']
                    if p_rank != v_rank:
                        for p_var in o_sdag.edges[(p,v)]['edge_variables']:
                            # s_var are variables computed from p that feeds into v
                            COMM_COST = ccl_graph.nodes[p_var]['cost']
                            schedule_pt2pt_comm(
                                dummy_schedule,
                                schedule_estimated_timeline = dummy_estimated_timeline,
                                node = p_var,
                                communication_cost= COMM_COST,
                                from_rank=p_rank,
                                to_rank=v_rank,

                            )

                # if v_rank != 0:
                #     for input_var in sdag.predecessors(v):
                #         if sdag.in_degree(input_var) == 0:
                #             schedule_pt2pt_comm(
                #                 dummy_schedule,
                #                 schedule_estimated_timeline = dummy_estimated_timeline,
                #                 node = input_var,
                #                 communication_cost= COMM_COST,
                #                 from_rank = 0,
                #                 to_rank = v_rank)
                            
                # compute start time of v and created idle
                v_start_time = dummy_estimated_timeline[v_rank][-1]
                idle_time = dummy_estimated_timeline[v_rank][-1] - dummy_estimated_timeline[v_rank][0]
                potential_v_schedules.append((v_start_time, idle_time, v_rank))

            # for each rank, sort by start time of v
            potential_v_schedules = sorted(potential_v_schedules)
            # print()
            # print(potential_v_schedules)
            DELTA = 0.0
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
            # print(best_rank)
            # ================== NEW ==================

            o_sdag.nodes[v]['rank'] = v_rank
            # v_rank = o_sdag.nodes[v]['rank']

            for p in o_sdag.predecessors(v):
                # p is an operation node
                p_rank = o_sdag.nodes[p]['rank']
                if p_rank != v_rank:
                    for p_var in o_sdag.edges[(p,v)]['edge_variables']:
                        # s_var are variables computed from p that feeds into v
                        COMM_COST = ccl_graph.nodes[p_var]['cost']
                        
                        schedule_pt2pt_comm(
                            schedule,
                            schedule_estimated_timeline = estimated_timeline,
                            node = p_var,
                            communication_cost= COMM_COST,
                            from_rank=p_rank,
                            to_rank=v_rank,
                        )

            # if v_rank != 0:
            #     for input_var in sdag.predecessors(v):
            #         if sdag.in_degree(input_var) == 0:
            #             schedule_pt2pt_comm(
            #                 schedule,
            #                 schedule_estimated_timeline = estimated_timeline,
            #                 node = input_var,
            #                 communication_cost= COMM_COST,
            #                 from_rank = 0,
            #                 to_rank = v_rank)

            # Schedule operation and all computed values
            OP_COST = ccl_graph.nodes[v]['cost']
            schedule_operation(
                schedule,
                schedule_estimated_timeline = estimated_timeline,
                operation = v,
                operation_cost=OP_COST,
                rank = v_rank
            )
            for s in o_sdag.successors(v):
                # If all input variables to s have been computed, s is available to compute
                o_sdag.nodes[s]['touches_left'] = o_sdag.nodes[s]['touches_left'] - 1
                if o_sdag.nodes[s]['touches_left'] == 0:
                    longest_time_ahead = o_sdag.nodes[s]['FWA']
                    heapq.heappush(available_operations_heap, (longest_time_ahead, s))
        

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
