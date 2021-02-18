
import networkx as nx
import numpy as np
from trecs.models import BassModel

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
        # because of the way BassModel is written, we are actually going to make
        # the edges have the source be the nodes following and the target
        # the nodes that are being followed
        G.add_edges_from(zip(in_nodes, np.ones(out_seq[i], dtype=int) * i))
    
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


# calculate metrics of interest
def popularity(simulation):
    return (simulation.infection_state.value != 0).sum()
    
def prob_large_cascade(sizes, pop_threshold=100):
    large_cascades = np.where(sizes > pop_threshold)[0]
    return len(large_cascades) / len(sizes)

def mean_virality(viralitys, popular_mask=None):
    # Assume virality of -1 are invalid trials
    # (i.e., the seed user was not able to infect)
    # any other user
    if popular_mask is None:
        return viralitys[viralitys > -1].mean()
    else: 
        # assume user passed in a mask of "popular" cascades to apply
        # first
        popular_viralitys = viralitys[popular_mask]
        return popular_viralitys[popular_viralitys > -1].mean()

def size_virality_corr(sizes, viralitys, only_popular=False, pop_threshold=100):
    """ Calculate correlation between size of cascade
        and structural virality of cascade. Only compute
        correlation on trials where >1 node was infected
        (and therefore structural virality is computable.)
    """
    if not only_popular:
        valid_sims = viralitys > -1 
    else:
        valid_sims = np.logical_and(viralitys > -1, sizes > pop_threshold) 
    stacked_obvs = np.vstack([sizes[valid_sims], viralitys[valid_sims]])
    return np.corrcoef(stacked_obvs)[0, 1]
