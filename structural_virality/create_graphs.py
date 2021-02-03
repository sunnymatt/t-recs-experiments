from scipy.sparse import csr_matrix, save_npz
import networkx as nx
import pickle as pkl
import os
import datetime

from graph_utils import calc_avg_degree, scale_free_graph

def stringify_alpha(alpha):
    """ Turns alpha float into a valid string
        for a subdirectory name.
    """
    alpha_string = f"alpha_{str(alpha).replace('.', '-')}"
    return alpha_string

if __name__ == "__main__":
    num_nodes = 100 # 1 million nodes per graph
    graph_dir = "graphs_1m"
    alphas = [2.1, 2.3, 2.5, 2.7, 2.9]
    num_graphs_per_alpha = 25

    # create graphs directory if one does not exist
    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)

    # iteratively create graphs
    for alpha in alphas:
        alpha_string = stringify_alpha(alpha)
        alpha_subdir = os.path.join(graph_dir, alpha_string)
        if not os.path.exists(alpha_subdir):
            os.mkdir(alpha_subdir)

        # for each level of alpha, create graphs and store in
        # separate subdirectory
        print(f"Creating graphs for alpha={alpha} at time {datetime.datetime.now()}...")
        for i in range(num_graphs_per_alpha):
            param_dict = {}
            subdir = os.path.join(alpha_subdir, str(i))
            if not os.path.exists(subdir):
                os.mkdir(subdir)
            G = scale_free_graph(num_nodes, alpha=alpha)

            print(f"\tFinished creating graph {i} at time {datetime.datetime.now()}...")
            user_rep = nx.convert_matrix.to_scipy_sparse_matrix(G) # convert to scipy adjacency matrix
            save_npz(os.path.join(subdir, "sparse_matrix.npz"), user_rep)
            param_dict["k"] = calc_avg_degree(G)
            param_dict["alpha"] = alpha
            param_dict["num_nodes"] = num_nodes
            f = open(os.path.join(subdir, "param.pkl"), 'wb')
            pkl.dump(param_dict, f, -1)
            f.close()
        print()

