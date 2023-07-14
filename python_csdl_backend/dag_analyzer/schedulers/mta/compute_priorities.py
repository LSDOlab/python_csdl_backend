import networkx as nx

def compute_furthest_weight_ahead(DAG):
    # Initialize a dictionary to keep track of the furthest weight ahead for each node
    furthest_weight_ahead = {}

    # Perform a topological sort of the DAG
    topological_order = list(nx.topological_sort(DAG))

    # Traverse the nodes in reverse topological order and add them to the ordered list
    # ordered_nodes = []
    for node in reversed(topological_order):
        # If the node has no children, its furthest weight ahead is its own weight
        if DAG.out_degree(node) == 0:
            furthest_weight_ahead[node] = DAG.nodes[node]['time_cost']
        # Otherwise, its furthest weight ahead is the maximum of its children's furthest weights ahead
        else:
            furthest_weight_ahead[node] = max(
                furthest_weight_ahead[child]+DAG.nodes[node]['time_cost'] for child in DAG.successors(node)
            )
        # Add the node to the ordered list based on its furthest weight ahead
        # index = len(ordered_nodes)
        # while index > 0 and furthest_weight_ahead[node] > furthest_weight_ahead[ordered_nodes[index-1]]:
        #     index -= 1
        # ordered_nodes.insert(index, node)

        DAG.nodes[node]['FWA'] = -furthest_weight_ahead[node]

def compute_shortest_weight_before(DAG):
    # Initialize a dictionary to keep track of the furthest weight ahead for each node
    furthest_weight_ahead = {}

    # Perform a topological sort of the DAG
    topological_order = list(nx.topological_sort(DAG))

    # Traverse the nodes in reverse topological order and add them to the ordered list
    # ordered_nodes = []
    for node in (topological_order):
        # If the node has no children, its furthest weight ahead is its own weight
        if DAG.in_degree(node) == 0:
            furthest_weight_ahead[node] = DAG.nodes[node]['time_cost']
        # Otherwise, its furthest weight ahead is the maximum of its children's furthest weights ahead
        else:
            furthest_weight_ahead[node] = max(
                furthest_weight_ahead[child]+DAG.nodes[node]['time_cost'] for child in DAG.predecessors(node)
            )
        # Add the node to the ordered list based on its furthest weight ahead
        # index = len(ordered_nodes)
        # while index > 0 and furthest_weight_ahead[node] > furthest_weight_ahead[ordered_nodes[index-1]]:
        #     index -= 1
        # ordered_nodes.insert(index, node)

        DAG.nodes[node]['FWA'] = furthest_weight_ahead[node]