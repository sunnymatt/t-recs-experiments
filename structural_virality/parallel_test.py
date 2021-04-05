from numba import njit, prange
import numpy as np
import scipy.sparse as sp
import timeit
from graph_utils import deg_seq, calc_avg_degree 
from create_graphs import stringify_alpha
import os
import datetime
import pickle as pkl
import errno
import argparse

# @njit(parallel=True)
def adj_pairs_parallel(seq, num_nodes):
    in_n = np.zeros(seq.sum())
    out_n = np.zeros(seq.sum())
    idxs = np.concatenate((np.array([0]), np.cumsum(seq)))
    gen = np.random.default_rng()
    for i in range(len(seq)):
        in_nodes_slice = gen.choice(num_nodes - 1, seq[i], replace=False)
        # avoid self loops
        in_nodes_slice[in_nodes_slice >= i] += 1
        out_nodes = np.ones(seq[i], dtype=np.int64) * i
        # because of the way BassModel is written, we are actually going to make
        # the edges have the source be the nodes following and the target
        # the nodes that are being followed
        in_n[idxs[i]:idxs[i+1]] = in_nodes_slice
        out_n[idxs[i]:idxs[i+1]] = out_nodes
    return in_n, out_n

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='create scale-free graphs')
    parser.add_argument('--graph_dir', type=str, default='graphs_25m')
    parser.add_argument('--num_nodes', type=int, default=25000000)
    parser.add_argument('--alphas', type=float, nargs='+', default=[2.1, 2.3, 2.5, 2.7, 2.9])
    parser.add_argument('--graph_ids', type=int, nargs='+', required=True)
    args = parser.parse_args()
    arg_dict = vars(args)


    print(f"STARTING GRAPH GENERATION NOW...", flush=True)
    for alpha in arg_dict["alphas"]:
        alpha_dir = os.path.join(arg_dict["graph_dir"], stringify_alpha(alpha))
        try:
            os.makedirs(alpha_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass
        for i in arg_dict["graph_ids"]:
            param_dict = {}
            subdir = os.path.join(alpha_dir, str(i))
            try:
                os.makedirs(subdir)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass
            print(f"Generating degree sequence at time {datetime.datetime.now()}...", flush=True)
            seq = deg_seq(arg_dict["num_nodes"], alpha=alpha)
            print(f"Created degree sequence of length {seq.shape}", flush=True)
            print(f"\tCreating graph at time {datetime.datetime.now()}...", flush=True)

            g = sp.csr_matrix((arg_dict["num_nodes"], arg_dict["num_nodes"]), dtype=bool)    # now randomly connect nodes to each other based on the out_seq
            
            chunk_size = 5000000
            num_chunks = arg_dict["num_nodes"] // chunk_size
            print(f"Chunk size: {chunk_size}, total_chunks: {num_chunks}", flush=True)
            for i in range(num_chunks):
                print(f"\t\tOn node {i * chunk_size} out of {arg_dict['num_nodes']} at time {datetime.datetime.now()}...", flush=True)
                seq_slice = seq[chunk_size * i : chunk_size * (i+1)]
                in_n, out_n = adj_pairs_parallel(seq_slice, arg_dict["num_nodes"]) 
                g[in_n, out_n] = 1
            
            print(f"\tFinished creating graph {i} at time {datetime.datetime.now()}...", flush=True)
            sp.save_npz(os.path.join(subdir, "sparse_matrix.npz"), g)
            param_dict["k"] = g.sum() / arg_dict["num_nodes"]
            print(f"Average degree is {param_dict['k']}", flush=True)
            param_dict["alpha"] = alpha
            param_dict["num_nodes"] = arg_dict["num_nodes"]
            f = open(os.path.join(subdir, "param.pkl"), 'wb')
            pkl.dump(param_dict, f, -1)
            f.close()
            print()
        
