from python_csdl_backend.dag_analyzer.schedulers.algorithm import Algorithm
from python_csdl_backend.dag_analyzer.graph_generator.sdag_to_op_only_graph import sdag_to_op_only_graph
from python_csdl_backend.dag_analyzer.schedulers.mta.compute_priorities import compute_furthest_weight_ahead, compute_shortest_weight_before
from python_csdl_backend.dag_analyzer.schedulers.scheduler_functions import schedule_pt2pt_comm, schedule_operation
from python_csdl_backend.dag_analyzer.utils import draw

class SYNC_BASE(Algorithm):
    def __init__(self, priority = 'furthest ahead') -> None:
        super().__init__()
        self.priority = priority

        if self.priority == 'furthest ahead':
            self.compute_priorities = compute_furthest_weight_ahead
        elif self.priority == 'shortest before':
            self.compute_priorities = compute_shortest_weight_before

class SYNC_POINTS(SYNC_BASE):
    
    def schedule(self, ccl_graph):
        NUM_RANKS = self.NUM_RANKS
        sdag = ccl_graph
        if self.create_plots:
            draw(sdag, title = 'images/SDAG')

        # Translate standard graph to operation-only graph
        o_sdag, total_weight = sdag_to_op_only_graph(sdag)
        if self.create_plots:
            draw(o_sdag, title = 'images/OSDAG')

        # with open(f'TEMP.pickle', 'wb') as f:
        #     import pickle
        #     pickle.dump(o_sdag, f)
        # with open(f'TEMP2.pickle', 'wb') as f:
        #     import pickle
        #     pickle.dump(sdag, f)
        # exit()
        # compute_furthest_weight_ahead(o_sdag)
        self.compute_priorities(o_sdag)

        # Schedule data structure:
        # schedule = [[], [], [], ... (NUMBER_OF_PARTS times)]
        preliminary_schedule = []
        preliminary_estimated_timeline = []
        for _ in range(NUM_RANKS):
            preliminary_schedule.append([])                
            preliminary_estimated_timeline.append([0])

        # Preprocessing
        import heapq
        available_operations_set = set() # A is a heap containing available nodes where heap root is the best to schedule
        input_operations = set()
        output_variables = set()
        input_variables = set()
        num_ops = 0
        # Find all available nodes
        for node in sdag.nodes:
            # longest_time_ahead = o_sdag.nodes[node]['FWA']
            # print(longest_time_ahead, node)
            if sdag.nodes[node]['type'] == 'operation':
                num_ops += 1
                o_sdag.nodes[node]['touches_left'] = o_sdag.in_degree(node)
                o_sdag.nodes[node]['touches_left_final'] = o_sdag.in_degree(node)
                o_sdag.nodes[node]['release_time'] = 'NOT_READY'

                if o_sdag.in_degree(node) == 0:
                    if node not in input_operations:
                        input_operations.add(node)
                        longest_time_ahead = o_sdag.nodes[node]['FWA']
                        # available_operations_set.add((longest_time_ahead, node))
                        available_operations_set.add(node)
                        # heapq.heappush(available_operations_set, (longest_time_ahead, node))
            
            if sdag.in_degree(node) == 0:
                # If a input leaf operation has a non-zero rank, we need to send the inputs from rank zero to the appropriate rank
                input_variables.add(node)

                # Also, initialize the available operations heap
                # op_nodes = sdag.successors(node)
                # for op in op_nodes:
                #     if op not in input_operations:
                #         input_operations.add(op)
                #         longest_time_ahead = o_sdag.nodes[op]['FWA']
                #         heapq.heappush(available_operations_set, (longest_time_ahead, op))

            if sdag.out_degree(node) == 0:
                output_variables.add(node)
                # If a output leaf operation has a non-zero rank, we need to send the outputs from the appropriate rank to rank zero
                sdag.nodes[node]['output'] = 1

        # Algorithm:
        # basically, what we do is fill up each processor until they are "balanced" at a certain point in time.
        # This "balance" point is what we call a synchronization point. 
        # All point to point communication is done at a specific synchronization point
        # We create a synchronization point when we cannot add schedule any more operations within this synchronization point.

        # Global data structures:
        # A synchronization point is defined as an integer sp starting from zero in increasing order of time.
        # For a synchronization point sp, we store a list of operations that are scheduled between sp-1 and sp
        # We need data structures for two way mappings between operations and synchronization points
        # o->sp: hashmap[operation] --> sp(o):int
        #   - sp(o) is synchronization point directly succeeding operation o
        #   - We can keep this as part of the graph data structure
        # sp->o: list[sp] --> hasmhmap with set of operations/comms that are scheduled between sp-1 and sp
        #   - List of lists called schedule

        # Implementation:
        # Intialize current synchronization point sp = 0

        

        # SynchronizationPoint class
        SynchronizationPoint = build_sync_point_class(NUM_RANKS)

        sync_schedule = [SynchronizationPoint(0), SynchronizationPoint(1)]
        current_sp = 1
        num_operations_added = 0
        while available_operations_set:
            
            # Set up operations for sync point
            # Use heap to get priority for operation ordering
            # Heapify slower than list traversal???...
            available_operations_temp = []
            for op in available_operations_set:
                longest_time_ahead = o_sdag.nodes[op]['FWA']
                heapq.heappush(available_operations_temp,(longest_time_ahead, op))
                # heapq.heappush(available_operations_temp,(sdag.nodes[op]['cost'], op))
            current_sync_point = sync_schedule[current_sp]
            scheduled_ops = set()
            
            # Perform actual scheduling for this sync point
            while available_operations_temp:
                v_information = heapq.heappop(available_operations_temp) # Get best node to schedule
                v  = v_information[1] # v is the current node to schedule

                # Check if we can fit v into current synchronization point
                # Conditions: 
                # 1) The operations in the current synchronization point must not be a predecessor of v
                # 2) Adding V to the current synchronization point must not unbalance load
                v_rank = current_sync_point.get_best_rank(
                    operation = v,
                    o_sdag = o_sdag,
                    sdag = sdag,
                    cost = sdag.nodes[v]['cost'],
                )

                # If successful, add operation
                if v_rank is not None:
                    # Add operation to schedule!!
                    current_sync_point.add_operation(
                        operation = v,
                        rank = v_rank,
                        cost = sdag.nodes[v]['cost'],
                    )

                    # Logistical stuff
                    scheduled_ops.add(v)
                    o_sdag.nodes[v]['rank'] = v_rank
                    o_sdag.nodes[v]['synchronization point'] = current_sp

                    # Add communication nodes to any synchronization points that need it
                    for p in o_sdag.predecessors(v):
                        # p is an operation node
                        p_rank = o_sdag.nodes[p]['rank']
                        if p_rank != v_rank:
                            for p_var in o_sdag.edges[(p,v)]['edge_variables']:
                                COMM_COST = sdag.nodes[p_var]['cost']
                                sync_index = o_sdag.nodes[p]['synchronization point']
                                sync_schedule[sync_index].add_communication(
                                    node = p_var,
                                    comm_cost = COMM_COST,
                                    from_rank = p_rank,
                                    to_rank = v_rank,
                                )

                    if v_rank != 0:
                        for p_var in sdag.predecessors(v):
                            if sdag.in_degree(p_var) == 0:
                                COMM_COST = sdag.nodes[p_var]['cost']
                                sync_index = 0
                                sync_schedule[sync_index].add_communication(
                                    node = p_var,
                                    comm_cost = COMM_COST,
                                    from_rank = 0,
                                    to_rank = v_rank,
                                )

                # Terminate sync point scheduling early if load balanced potentially?
                if current_sync_point.load_balanced:
                    break

            # Now remove all scheduled operations
            for op in scheduled_ops:
                available_operations_set.remove(op)

                for s in o_sdag.successors(op):
                    # If all input variables to s have been computed, s is available to compute
                    o_sdag.nodes[s]['touches_left'] = o_sdag.nodes[s]['touches_left'] - 1
                    if o_sdag.nodes[s]['touches_left'] == 0:
                        available_operations_set.add(s)

            num_operations_added += len(scheduled_ops)
            # print(f"sync point {current_sp} ({num_operations_added}/{num_ops}) ({len(available_operations_set)})")
            # print(sync_schedule[-1].schedule)

            # Finished synced point
            current_sp += 1
            sync_schedule.append(SynchronizationPoint(current_sp))


        # Post Processing:
        # Send all output variables to rank 0
        for output_var in output_variables:
            preds = list(sdag.predecessors(output_var))
            if len(preds) > 0:
                op = preds[0]
                op_rank = o_sdag.nodes[op]['rank']
                if op_rank != 0:
                    COMM_COST = sdag.nodes[output_var]['cost']
                    sync_schedule[-1].add_communication(
                        node = output_var,
                        comm_cost = COMM_COST,
                        from_rank = op_rank,
                        to_rank = 0,
                    )

        # FINAL SCHEDULE BUILD
        # initialize data structures
        schedule = []
        estimated_timeline = []
        for _ in range(NUM_RANKS):
            schedule.append([])                
            estimated_timeline.append([0])
        
        for sync_point in sync_schedule:
            for operation in sync_point.all_operations_ordered:
                rank = o_sdag.nodes[operation]['rank']
                OP_COST = sdag.nodes[operation]['cost']
                schedule_operation(
                    schedule,
                    schedule_estimated_timeline = estimated_timeline,
                    operation = operation,
                    operation_cost=OP_COST,
                    rank = rank
                )
            for communication in sync_point.all_comms_ordered:
                node = communication[0]
                comm_cost = communication[1]
                from_rank = communication[2]
                to_rank = communication[3]
                schedule_pt2pt_comm(
                    schedule,
                    schedule_estimated_timeline = estimated_timeline,
                    node = node,
                    communication_cost= comm_cost,
                    from_rank=from_rank,
                    to_rank=to_rank,
                )
        # import time
        # time.sleep(0.2)
        # raise ValueError('UNFINISHED ALGORITHM :(')
        return schedule, estimated_timeline
        

def build_sync_point_class(NUM_RANKS):
    class SynchronizationPoint():

        def __init__(
                self, 
                sp: int,
                ):
            # Integeter of current synchronization point
            self.id = sp
            self.load_balanced = False
            self.max_ratio = 1.5 # tuning parameter
            self.min_load_imbalance_cutoff = 0.1 # tuning parameter
            self.max_load_balance_cutoff = 0.8 # tuning parameter

            # Length of sync point interval
            self.interval_length = 0
            self.comm_length = 0
            self.max_length = 0
            self.all_operations = set()
            self.all_operations_ordered = []
            self.all_comms_ordered = []
            self.all_comms = set()

            # List of lists called schedule
            # all operations that are scheduled between sp-1 and sp
            self.schedule = []
            self.schedule_lengths = []
            # self.communications = []
            self.open_ranks = set()
            for i in range(NUM_RANKS):
                self.schedule.append(set())
                self.schedule_lengths.append(0)
                # self.communications.append(set())
                self.open_ranks.add(i)

        def add_operation(self, operation, rank, cost):
            # Add operation to schedule
            self.schedule[rank].add(operation)
            self.all_operations.add(operation)
            self.all_operations_ordered.append(operation)

            # Update schedule length
            new_cost = self.schedule_lengths[rank] + cost
            self.schedule_lengths[rank] = new_cost
            if new_cost > self.interval_length:
                self.interval_length = new_cost

            # Update open ranks
            if rank in self.open_ranks:
                self.open_ranks.remove(rank)

        def add_communication(
                self,
                node,
                comm_cost,
                from_rank,
                to_rank,
            ):
            communication = (node, comm_cost, from_rank, to_rank)
            # self.communications[rank].add(communication)
            if communication not in self.all_comms:
                # self.communications[rank].add(communication)
                self.all_comms_ordered.append(communication)
                self.all_comms.add(communication)

        
        def compute_load_balance_metric(self, rank_add, cost_add):

            sync_interval_time = 0
            for rank, schedule_length in enumerate(self.schedule_lengths):
                if rank == rank_add:
                    sync_interval_time = max(sync_interval_time, schedule_length + cost_add)
                else:
                    sync_interval_time = max(sync_interval_time, schedule_length)
            
            percent_idle = 0
            for rank, schedule_length in enumerate(self.schedule_lengths):
                if rank == rank_add:
                    percent_idle += (sync_interval_time - (schedule_length + cost_add))/sync_interval_time/NUM_RANKS
                else:
                    percent_idle += (sync_interval_time - schedule_length)/sync_interval_time/NUM_RANKS
            return percent_idle


        def get_best_rank(
                self,
                operation, #Operation to schedyle
                o_sdag, # Operation only sdag
                sdag, # Full sdag
                cost, # Cost of operation
            ):
            """
            returns the best rank (integer) for this operation.
            If there is no available rank, return None
            """

            # Predecessors of operation cannot be in the current synchronization point
            # if cost > 1e-3:
            #     print('EXPENSIVE OP', cost, self.id, len(self.open_ranks))
            pred_ranks = set() 
            for pred in o_sdag.predecessors(operation):
                if pred in self.all_operations and (len(pred_ranks) > 1):
                    return None
                
                for p_var in o_sdag.edges[(pred,operation)]['edge_variables']:
                    # s_var are variables computed from p that feeds into v
                    COMM_COST = sdag.nodes[p_var]['cost']
                pred_ranks.add((o_sdag.nodes[pred]['rank'], COMM_COST))

                # if len(pred_ranks) > 1:
                #     return None

            # Find the best rank
            for rank in range(NUM_RANKS):
                # compute communication costs
                communication_costs = 0
                for pred_rank in pred_ranks:
                    if pred_rank[0] == rank:
                        continue
                    communication_costs += pred_rank[1]
                
                # Find best rank that minimizes start time
                if rank == 0:
                    best_rank = 0
                    best_start_time = self.schedule_lengths[rank] + communication_costs
                else:
                    this_rank_start_time = self.schedule_lengths[rank] + communication_costs
                    # If this rank if better, update it
                    if this_rank_start_time < best_start_time:
                        best_rank = rank
                        best_start_time = this_rank_start_time

            # Now we need to check if adding this operation to the best rank will 
            # increase length by 0.5
            if best_rank in self.open_ranks:
                self.max_length = max(self.interval_length, self.max_length, best_start_time + cost)
                return best_rank
            else:
                # load_balance_metric = self.compute_load_balance_metric(best_rank, cost)
                # if load_balance_metric < self.min_load_imbalance_cutoff:
                #     print('LOAD IMBALANCE:', load_balance_metric, cost)
                #     return None
                # elif load_balance_metric > self.max_load_balance_cutoff:
                #     print('LOAD BALANCED:', load_balance_metric, cost)
                #     self.load_balanced = True
                #     return best_rank

                # self.load_balanced = True
                # We cannot accept this operation if interval length increases by t0o much
                if (best_start_time + cost) > (self.max_length*self.max_ratio):

                    return None
                else:
                    return best_rank
                
    return SynchronizationPoint