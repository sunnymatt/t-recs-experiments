# Matthew Sun, 2/2/2021
# Running structural virality simulations from premade graphs in a particular folder

from guppy import hpy
from collections import defaultdict
import datetime
import multiprocessing as mp
import os
import pickle as pkl
import sys
from scipy.sparse import load_npz
import numpy as np
from trecs.models import BassModel
from create_graphs import stringify_alpha
from graph_utils import setup_experiment, popularity

    
def run_sims(alpha, r, sims_per_graph, graph_dir):
    """ Runs set of simulations on a set of premade graphs
        for a given level of alpha and r on one particular
        graph. A total of sims_per_graph simulations
        will be run. It then places the resulting
        sizes / SV values as a dictionary in the out_q Queue.
    """
    # these will store results
    size_arr = np.zeros(sims_per_graph)
    vir_arr = np.zeros(sims_per_graph)
    # keep track of index of trial
    trial_idx = 0
    
    # find subfolders for each individual graph
    user_rep = load_npz(os.path.join(graph_dir, "sparse_matrix.npz"))
    print(user_rep.shape)
    # param_dict contains k, r, beta, and num_nodes
    param_dict = pkl.load(open(os.path.join(graph_dir, "param.pkl"), "rb")) 
    
    print(f"alpha={alpha}, r={r}: Starting {sims_per_graph} simulations on graph in {graph_dir}...")
    for j in range(sims_per_graph):
        simulation = setup_experiment(user_rep, param_dict["k"], r=r)
        simulation.run()
        size = popularity(simulation)
        size_arr[trial_idx] = size
        try:
            sv = simulation.get_structural_virality()
            vir_arr[trial_idx] = sv
        except:
            vir_arr[trial_idx] = -1 # couldn't calculate virality
        trial_idx += 1
    print(f"alpha={alpha}, r={r}: Completed {sims_per_graph} simulations on graph in {graph_dir}!")
    
if __name__ == "__main__":
    h = hpy()
     
    run_sims(2.1, 0.5, 10, "graphs_1m/alpha_2-1/0")
    print(h.heap())
    print()
    print(f"Total memory: {h.heap().size}")
