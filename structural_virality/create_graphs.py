from scipy.sparse import csr_matrix, save_npz
import networkx as nx
import pickle as pkl
import os
import datetime

from graph_utils import calc_avg_degree, scale_free_graph

NUM_NODES = 100000
OUTPUT_DIR = "graphs_100k"
ALPHAS = [2.1, 2.3, 2.5, 2.7, 2.9] # generate graphs with degrees governed by different distributions
NUM_GRAPHS_PER_ALPHA = 25

def stringify_alpha(alpha):
    """ Turns alpha float into a valid string
        for a subdirectory name.
    """
    alpha_string = f"alpha_{str(alpha).replace('.', '-')}"
    return alpha_string

if __name__ == "__main__":
    """ Creates a "graphs" folder with graphs stored in the following structure:
        
        graphs_100k/ # name set equal to constant at top of file
        |___ alpha_2-1/                   # corresponds to alpha = 2.1
              |__ 0/                      # graph number 0
                  |__ sparse_matrix.npz   # saved version of matrix
                  |__ param.pkl           # stores info like # of nodes, k (average degree), and alpha
              |__ 1                       # graph number 1
                  |__ sparse_matrix.npz
                  |__ param.pkl
              |__ ..
        |___ alpha_2-3                    # more folders for other values of alpha
        |___ ...
    """
    
    # create graphs directory if one does not exist
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # iteratively create graphs
    for alpha in ALPHAS:
        alpha_string = stringify_alpha(alpha)
        alpha_subdir = os.path.join(OUTPUT_DIR, alpha_string)
        if not os.path.exists(alpha_subdir):
            os.mkdir(alpha_subdir)

        # for each level of alpha, create graphs and store in
        # separate subdirectory
        print(f"Creating graphs for alpha={alpha} at time {datetime.datetime.now()}...")
        for i in range(NUM_GRAPHS_PER_ALPHA):
            param_dict = {}
            subdir = os.path.join(alpha_subdir, str(i))
            if not os.path.exists(subdir):
                os.mkdir(subdir)
            G = scale_free_graph(NUM_NODES, alpha=alpha)

            print(f"\tFinished creating graph {i} at time {datetime.datetime.now()}...")
            user_rep = nx.convert_matrix.to_scipy_sparse_matrix(G) # convert to scipy adjacency matrix
            save_npz(os.path.join(subdir, "sparse_matrix.npz"), user_rep)
            param_dict["k"] = calc_avg_degree(G)
            param_dict["alpha"] = alpha
            param_dict["num_nodes"] = NUM_NODES
            f = open(os.path.join(subdir, "param.pkl"), 'wb')
            pkl.dump(param_dict, f, -1)
            f.close()
        print()

