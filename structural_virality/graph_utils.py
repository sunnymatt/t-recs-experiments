import networkx as nx
import numpy as np

def calc_avg_degree(graph):
    """ Assumes directed graph, where each edge only counts
        for 1 degree (for the source node, not the target node).
    """
    sum_of_edges = sum(deg for n, deg in graph.degree())
    k = sum_of_edges / graph.number_of_nodes()
    return k / 2 # divide by two 

def implied_beta(k, r):
    """ Formula: r = k * beta, so beta equals r/k.
    """
    return r / k

def scale_free_graph(num_nodes, alpha=2.3):
    """ Generate the scale free graph with the degree sequence specified
        by the power law distribution parameterized by alpha. 
    """
    min_edges = 10
    max_edges = min(num_nodes - 1, 1e6)
    out_seq = np.zeros(num_nodes, dtype=int)
    
    idx = 0
    # following the paper, the out-degree is specified by a power law sequence
    while idx < num_nodes:
        power_seq = np.array(nx.utils.powerlaw_sequence(num_nodes - idx, alpha)).astype(int)
        # filter to edge range
        power_seq = power_seq[np.logical_and(power_seq >= min_edges, power_seq <= max_edges)]
        end_idx = idx + len(power_seq)
        if end_idx == num_nodes:
            if (out_seq.sum() + power_seq.sum()) % 2 != 0: # must have even total out-degree
                continue
        
        out_seq[idx:end_idx] = power_seq
        idx = end_idx

    G = nx.DiGraph()
    # now randomly connect nodes to each other based on the out_seq
    # we sample other nodes without replacement
    rng = np.random.default_rng()
    for i in range(len(out_seq)):
        # only connect to the nodes besides this node
        # no duplicate connections or self loops
        in_nodes = rng.choice(num_nodes - 1, out_seq[i], replace=False)
        in_nodes[in_nodes >= i] +=1 # avoid self loops
        G.add_edges_from(zip(np.ones(out_seq[i], dtype=int) * i, in_nodes))
    
    return G

def setup_experiment(user_rep, k, r=0.5):
    beta = implied_beta(k, r)
    item_rep = np.array([[beta]]) # must be two dimensional
    
    # seed infection with 1 user
    num_users = user_rep.shape[0]
    infection_state = np.zeros(num_users).reshape(-1, 1) # must be two dimensional array
    seed_user = np.random.choice(num_users, 1)
    infection_state[seed_user, 0] = 1
    
    # create model
    bass = BassModel(
        user_representation=user_rep,
        item_representation=item_rep,
        infection_state=infection_state
    )
    return bass