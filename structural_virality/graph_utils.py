import networkx as nx
import numpy as np

def calc_avg_degree(graph):
    sum_of_edges = sum(deg for n, deg in graph.degree())
    k = sum_of_edges / graph.number_of_nodes()
    return k

def implied_beta(k, r):
    """ Formula: r = k * beta, so beta equals r/k.
    """
    return r / k

def scale_free_graph(num_nodes, alpha=2.3):
    """ Generate the scale free graph with the degree sequence specified
        by the power law distribution parameterized by alpha. 
    """
    int_seq = np.zeros(num_nodes).astype(int)
    idx = 0
    while idx < num_nodes:
        nextval = int(nx.utils.powerlaw_sequence(1, alpha)[0])
        if idx == num_nodes - 1 and (int_seq.sum() + nextval) % 2 != 0 : # make sure sum is even
            continue
        if nextval > 10 and nextval < 1e6:
            int_seq[idx] = nextval
            idx += 1
    # can't use trecs.SocialGraphGenerator because "n" is not an argument to configuration_model
    # remove self-loops and duplicate edges
    G = nx.configuration_model(int_seq) 
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G