from scipy.sparse import csr_matrix, save_npz
import networkx as nx
import pickle as pkl
import os
import datetime
import errno
import argparse
from graph_utils import calc_avg_degree, scale_free_graph

def stringify_alpha(alpha):
    """ Turns alpha float into a valid string
        for a subdirectory name.
    """
    alpha_string = f"alpha_{str(alpha).replace('.', '-')}"
    return alpha_string

def stringify_r(r):
    """ Turns r float into a valid string
        for a subdirectory name.
    """
    r_string = f"r_{str(r).replace('.', '-')}"
    return r_string

def destringify_alpha(alpha_string):
    """ Extracts alpha float from a valid string
        for a subdirectory name.
    """
    alpha_string = alpha_string[6:].replace('-', '.')
    return float(alpha_string)

def destringify_r(r_string):
    """ Extracts r float from a valid string
        for a subdirectory name.
    """
    r_string = r_string[2:].replace('-', '.')
    return float(r_string)


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
    # parse arguments
    parser = argparse.ArgumentParser(description='create scale-free graphs')
    parser.add_argument('--graph_dir', type=str, default='graphs_1m')
    parser.add_argument('--num_nodes', type=int, default=1000000)
    parser.add_argument('--num_graphs_per_alpha', type=int, default=25)
    parser.add_argument('--alphas', type=float, nargs='+', default=[2.1, 2.3, 2.5, 2.7, 2.9])
    args = parser.parse_args()
    arg_dict = vars(args)    
    # create graphs directory if one does not exist
    try:
        os.makedirs(arg_dict["graph_dir"])
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    # iteratively create graphs
    for alpha in arg_dict["alphas"]:
        alpha_string = stringify_alpha(alpha)
        alpha_subdir = os.path.join(arg_dict["graph_dir"], alpha_string)
        try:
            os.makedirs(alpha_subdir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

        # for each level of alpha, create graphs and store in
        # separate subdirectory
        print(f"Creating graphs for alpha={alpha} at time {datetime.datetime.now()}...")
        for i in range(arg_dict["num_graphs_per_alpha"]):
            param_dict = {}
            subdir = os.path.join(alpha_subdir, str(i))
            try:
                os.makedirs(subdir)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass
            G = scale_free_graph(arg_dict["num_nodes"], alpha=alpha)

            print(f"\tFinished creating graph {i} at time {datetime.datetime.now()}...")
            save_npz(os.path.join(subdir, "sparse_matrix.npz"), user_rep)
            param_dict["k"] = calc_avg_degree(G)
            param_dict["alpha"] = alpha
            param_dict["num_nodes"] = arg_dict["num_nodes"]
            f = open(os.path.join(subdir, "param.pkl"), 'wb')
            pkl.dump(param_dict, f, -1)
            f.close()
        print()

