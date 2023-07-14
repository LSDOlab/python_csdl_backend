from python_csdl_backend.dag_analyzer.schedulers.algorithm import Algorithm
from python_csdl_backend.dag_analyzer.graph_generator.sdag_to_op_only_graph import sdag_to_op_only_graph

from python_csdl_backend.dag_analyzer.schedulers.mta.compute_priorities import compute_furthest_weight_ahead
from python_csdl_backend.dag_analyzer.schedulers.scheduler_functions import schedule_pt2pt_comm, schedule_operation

from python_csdl_backend.dag_analyzer.utils import draw
from copy import deepcopy
class MTA_ETA_SLOW(Algorithm):
    
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

        num_nodes = len(o_sdag)
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

        
        # Main loop:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        print('start')
        current_iter = 0
        while available_operations_heap:
            current_iter += 1
            v_information = heapq.heappop(available_operations_heap) # Get best node to schedule
            v  = v_information[1] # v is the current node to schedule
            print('\t', v , f'\t({current_iter}/{num_nodes})')

            # Finding Best Rank Algorithm: (This is the important part)
            # record end time of all ranks as of right now
            # For each rank:
            #   For each pred of v in order of end_time:
            #       if the predecessor is on rank(v):     
            #           do not to anything, continue
            #       simulate a point to point from rank(pred) to rank(v)
            #       preprocessing:
            #       - *find minimum end time of rank(v) (we call this operation the blocker) that is greater than end_time(pred)
            #       to simulate, there are two cases:
            #       1) if end_time(pred) < end_time(blocker)
            #       - **shift end time of ALL upstream operations of rank(pred) by (end_time(blocker) - end_time(pred))
            #       2) if end_time(pred) > end_time(blocker)
            #       - ***shift end time of ALL upstream operations of rank(v) by (end_time(pred) - end_time(blocker))

            #       -----------------------illustration of 1)-----------------------:
            #       rank(v):   ____________________________ _ __ __ __ __ __ 
            #       rank 1:   |___|____blocker____|________|_ _potential v__|
            #                                ^
            #                                |
            #   rank(pred):    ______________|________
            #       rank 2:   |_____|__pred__|________|
            #
            #       -----------------------illustration of 2)-----------------------:
            #       rank(v):   _______________ _ __ __ __ __ __ 
            #       rank 1:   |___|__blocker__|_ _potential v__|
            #                                      ^
            #                                      |
            #   rank(pred):    ____________________|________
            #       rank 2:   |_____|_____pred_____|________|
            #       ----------------------------------------------------------------:

            #   record new end times of all ranks
            # Depending on all new end times, find the best rank to schedule v on

            # The stars represent expensive parts of the algorithm
            # * How do I find the blocker efficiently? brute force through all operations until I find the correct operation?
            #   binary search will reduce to log time but still expensive.
            # ** I don't see any other alternative than a loop through all upstream operations
            #    maybe store all operations in a vector and slice all operations and add difference there?
            # *** Same as **

            # Start algorithm:
            current_rank_end_times = []
            for v_rank in range(NUM_RANKS):
                current_rank_end_times.append(preliminary_estimated_timeline[v_rank][-1])

            v_rank_costs = []
            potential_rank = None
            potential_rank_cost= None
            potential_rank_info = {}
            for v_rank in range(NUM_RANKS):

                proposed_rank_end_times = deepcopy(current_rank_end_times)

                rank_info = {
                    'vars to communicate': [], # v2c[i] is a subset of predecessors of v that are on rank i 
                    'comm indices': [], # ci[i] is the index to actually perform communication so push upstream operation release times
                    'diffs': [], # time to push everything after comm_indices[i] by
                }

                for p in o_sdag.predecessors(v):
                    # p is an operation
                    p_rank = o_sdag.nodes[p]['rank']

                for p in o_sdag.predecessors(v):
                    
                    # p is an operation
                    p_rank = o_sdag.nodes[p]['rank']

                    # skip if p is on the same rank as v
                    if p_rank == v_rank:
                        continue
                    # Find the minimum time needed to schedule v on v_rank
                    for p_var in o_sdag.edges[(p,v)]['edge_variables']:
                        # s_var are variables computed from p that feeds into v
                        COMM_COST = ccl_graph.nodes[p_var]['cost']
                    
                    diff, p_rank_index = get_diff(o_sdag, ccl_graph, p, p_rank, v, v_rank, preliminary_schedule)
                    if diff > 0:
                        proposed_rank_end_times[p_rank] += (diff + COMM_COST)
                    else:
                        proposed_rank_end_times[v_rank] += (-diff + COMM_COST )
                
                cur_cost = 0
                for vv_rank in range(NUM_RANKS):
                    cur_cost += proposed_rank_end_times[vv_rank] - current_rank_end_times[vv_rank]
                v_rank_costs.append(cur_cost)

                if potential_rank is None:
                    potential_rank = v_rank
                    potential_rank_cost = cur_cost
                else:
                    if cur_cost < potential_rank_cost:
                        potential_rank = v_rank
                        potential_rank_cost = cur_cost
            
            v_rank = potential_rank
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
            # global_operation_ordering.append(v)
            # Update heap
            for s in o_sdag.successors(v):
                # If all input variables to s have been computed, s is available to compute
                o_sdag.nodes[s]['touches_left'] = o_sdag.nodes[s]['touches_left'] - 1
                if o_sdag.nodes[s]['touches_left'] == 0:
                    longest_time_ahead = o_sdag.nodes[s]['FWA']
                    heapq.heappush(available_operations_heap, (longest_time_ahead, s))
        
        profiler.disable()
        profiler.dump_stats('output')

        raise SystemExit('UNFINISHED ALGORITHM :(')
        exit('UNFINISHED ALGORITHM :(')
        return schedule, estimated_timeline


def get_diff(o_sdag, ccl_graph, p, p_rank, v, v_rank, preliminary_schedule):

    # print(p)
    # simulate a point to point from rank(p) to rank(v)
    # preprocessing:
    for i, op in enumerate(preliminary_schedule[p_rank]):
        # if op not in o_sdag.nodes:
        #     continue
        blocker = op
        blocker_end_time = o_sdag.nodes[op]['release_time']
        if blocker_end_time > o_sdag.nodes[p]['release_time']:
            break
    
    # if the blocker operation is
    diff = blocker_end_time - o_sdag.nodes[p]['release_time']
    return diff, i