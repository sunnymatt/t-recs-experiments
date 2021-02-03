# Matthew Sun, 2/2/2021
# Running structural virality simulations from premade graphs in a particular folder

from collections import defaultdict
import datetime
import os
import pickle as pkl
from scipy.sparse import load_npz
import numpy as np
from trecs.models import BassModel
from create_graphs import stringify_alpha
from graph_utils import setup_experiment, popularity

GRAPH_DIR = "graphs_1m"
SIMS_PER_GRAPH = 10
RESULTS_FILENAME = "sv_sims_1m_nodes.pkl"

# check folders that are supposed to exist actually do exist
def check_alpha_folders(alphas):
    alpha_to_dirname = {}
    for alpha in alphas:
        alpha_subdir = os.path.join(GRAPH_DIR, stringify_alpha(alpha))
        # check to ensure subdirectory exists
        if not os.path.exists(alpha_subdir):
            print("Uh oh! Cannot find graphs related to alpha={alpha}. Moving on...")
            continue
        alpha_to_dirname[alpha] = alpha_subdir
        # count how many graphs were made with this level of alpha
        graph_subdirs = [f.name for f in os.scandir(alpha_subdir) if f.is_dir()]
        print(f"Available graph directories for alpha={alpha}: {', '.join(graph_subdirs)}")
        print(f"Total graphs available: {len(graph_subdirs)}")
        print()
    return alpha_to_dirname
        
def save_results(results, filename):
    # save results with popualrity & structural virality for every
    # simulation at each level of alpha & r to a pickle file
    f = open(filename, "wb")
    pkl.dump(results, f, -1)
    f.close()
    
def run_sims(alpha, r, sims_per_graph, alpha_dir_map):
    """ Runs set of simulations on a set of premade graphs
        for a given level of alpha and r. A total of
        (sims_per_graph) * (number of graphs in directory) simulations
        will be run.
    """
    alpha_subdir = alpha_dir_map[alpha]
    graph_subdirs = [f.name for f in os.scandir(alpha_subdir) if f.is_dir()]
    
    total_trials = len(graph_subdirs) * SIMS_PER_GRAPH
    # these will store results
    size_arr = np.zeros(total_trials)
    vir_arr = np.zeros(total_trials)
    # keep track of index of trial
    trial_idx = 0
    
    # find subfolders for each individual graph
    for i, graph_subdir in enumerate(graph_subdirs):
        user_rep = load_npz(os.path.join(alpha_subdir, graph_subdir, "sparse_matrix.npz"))
        # param_dict contains k, r, beta, and num_nodes
        param_dict = pkl.load(open(os.path.join(alpha_subdir, graph_subdir, "param.pkl"), "rb")) 

        if i % 10 == 0:
            print(f"\tOn graph {i} at time {datetime.datetime.now()}...")

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
    return size_arr, vir_arr
    
if __name__ == "__main__":
    results = defaultdict(dict)
    # varying alpha and R
    alphas = [2.1, 2.3, 2.5, 2.7, 2.9]
    rs = [0.1, 0.3, 0.5, 0.7, 0.9]
    alpha_dir_map = check_alpha_folders(alphas)

    for alpha in alphas:
        for r in rs:
            print(f"Testing pair of parameters alpha={alpha}, r={r} at time {datetime.datetime.now()}...")
            size_arr, vir_arr = run_sims(alpha, r, SIMS_PER_GRAPH, alpha_dir_map)
            results[(alpha, r)]["size"] = size_arr
            results[(alpha, r)]["virality"] = vir_arr                  
            print("")
    
    save_results(results, RESULTS_FILENAME)
