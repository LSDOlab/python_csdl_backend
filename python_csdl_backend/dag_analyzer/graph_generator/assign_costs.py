def assign_costs(
        ccl_graph,
        communication_cost,
        operation_cost,
    ):

    for node in ccl_graph.nodes:
        if ccl_graph.nodes[node]['type'] == 'variable':
            ccl_graph.nodes[node]['cost'] = communication_cost
        elif ccl_graph.nodes[node]['type'] == 'operation':
            ccl_graph.nodes[node]['cost'] = operation_cost
            ccl_graph.nodes[node]['time_cost'] = operation_cost
        else:
            raise KeyError('EKJRNEKN')

def assign_heterogenous_costs(
        graph,
        communication_cost,
        ratios: list, # Ratio of operation cost compared to baseline cost
        chance: list, # % chance of operation cost for each ratio
    ):
    import random
    random.seed(9001)
    """
    Assign heterogenous costs to operations

    For example, if ratios = [10, 100] and chance = [0.1, 0.01], then 
    10% chance operation will have a cost of 10x the baseline cost, 
    and 1% chance operations will have a cost of 100x the baseline cost
    """

    # Check that ratios and chance are the same length
    if len(ratios) != len(chance):
        raise ValueError('ratios and chance must be the same length')
    
    # Check that chance sums to 1
    if sum(chance) > 1:
        raise ValueError('chances sum greater than 1')    
    baseline_cost = 1.0
    ratios.append(baseline_cost)
    chance.append(1 - sum(chance))

    for node in graph:
        if graph.nodes[node]['type'] == 'variable':
            graph.nodes[node]['cost'] = communication_cost
        elif graph.nodes[node]['type'] == 'operation':
            # Generate a random number between 0 and 1
            random_number = random.random()
            cumulative_chance = 0

            for ratio, probability in zip(ratios, chance):
                cumulative_chance += probability

                if random_number <= cumulative_chance:
                    # Assign cost to the node based on the ratio
                    operation_cost = baseline_cost * ratio
                    graph.nodes[node]['cost'] = operation_cost
                    graph.nodes[node]['time_cost'] = operation_cost
                    break
        else:
            raise KeyError('EKJRNEKN')
        
    for node in graph:
        if graph.nodes[node]['type'] == 'operation':
            random_number = ((random.random()*10)+75)/100
            graph.nodes[node]['cost'] = graph.nodes[node]['cost']*random_number
            graph.nodes[node]['time_cost'] = graph.nodes[node]['cost']
def normalize_costs(
        ccl_graph,
        max_time, # in seconds
        communication_cost,
    ):
    """
    normalizes time for each operation so that the total time does not exceed max_time
    """

    # Calculate total time
    total_time = 0.0
    for node in ccl_graph.nodes:
        if ccl_graph.nodes[node]['type'] == 'operation':
            total_time += ccl_graph.nodes[node]['time_cost']
    normailization_factor = max_time/total_time

    # assign costs
    for node in ccl_graph.nodes:
        if ccl_graph.nodes[node]['type'] == 'variable':
            ccl_graph.nodes[node]['cost'] = communication_cost
        elif ccl_graph.nodes[node]['type'] == 'operation':
            ccl_graph.nodes[node]['cost'] = ccl_graph.nodes[node]['time_cost']*normailization_factor
            ccl_graph.nodes[node]['time_cost'] = ccl_graph.nodes[node]['time_cost']*normailization_factor
        else:
            raise KeyError('EKJRNEKN')