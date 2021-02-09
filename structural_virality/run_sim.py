# Matthew Sun, 2/2/2021
# Running structural virality simulations from premade graphs in a particular folder

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

GRAPH_DIR = "graphs_10k"
SIMS_PER_GRAPH = 1
RESULTS_FILENAME = "test.pkl"
OUTPUT_DIR = "graphs_10k_sim_test"
LOG_PATH = os.path.join(OUTPUT_DIR, "log.txt")
PARALLELIZE = True
MAX_CPU_COUNT = 25
OUT_Q = mp.Queue() # this is what we'll use to update results 

# check folders that are supposed to exist actually do exist
# and create intermediate output folders
def check_alpha_folders(graph_dir, output_dir, alphas):
    alpha_to_dirname = {}
    alpha_to_graphs = {}
    print(f"Looking in directory {graph_dir} for graphs...")
    for alpha in alphas:
        alpha_string = stringify_alpha(alpha)
        alpha_subdir = os.path.join(graph_dir, alpha_string)
        # check to ensure subdirectory exists
        if not os.path.exists(alpha_subdir):
            print("Uh oh! Cannot find graphs related to alpha={alpha}. Moving on...")
            continue
        output_alpha_dir = os.path.join(output_dir, alpha_string)
        if not os.path.exists(output_alpha_dir):
            os.makedirs(output_alpha_dir)
        alpha_to_dirname[alpha] = alpha_subdir
        # count how many graphs were made with this level of alpha
        graph_subdirs = [f.name for f in os.scandir(alpha_subdir) if f.is_dir()]
        # make graph output dirs
        for g in graph_subdirs:
            graph_outdir = os.path.join(output_alpha_dir, g)
            if not os.path.exists(graph_outdir):
                os.makedirs(graph_outdir)
        alpha_to_graphs[alpha] = graph_subdirs
        print(f"Available graph directories for alpha={alpha}: {', '.join(graph_subdirs)}")
        print(f"Total graphs available: {len(graph_subdirs)}")
        print()
    return alpha_to_dirname, alpha_to_graphs
        
def save_results(results, filename):
    # save results with popualrity & structural virality for every
    # simulation at each level of alpha & r to a pickle file
    f = open(filename, "wb")
    pkl.dump(results, f, -1)
    f.close()
    
def print_to_log(msg, lock):
    lock.acquire()
    f = open(LOG_PATH, "a+")
    print(msg, file=f, flush=True)
    f.close()
    lock.release()

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
    # param_dict contains k, r, beta, and num_nodes
    param_dict = pkl.load(open(os.path.join(graph_dir, "param.pkl"), "rb")) 
    
    print_to_log(f"alpha={alpha}, r={r}: Starting {sims_per_graph} simulations on graph in {graph_dir} at time {datetime.datetime.now()} ...", LOCK)
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
    print_to_log(f"alpha={alpha}, r={r}: Completed {sims_per_graph} simulations on graph in {graph_dir}! at time {datetime.datetime.now()} ", LOCK)
    OUT_Q.put((alpha, r, size_arr, vir_arr, graph_dir))
   

def init(lock):
    """ So processes can write to the same file """
    global LOCK
    LOCK = lock
 
if __name__ == "__main__":
    results = {}
    # varying alpha and R
    alphas = [2.1, 2.3, 2.5, 2.7, 2.9]
    rs = [0.1, 0.3, 0.5, 0.7, 0.9]
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    alpha_dir_map, alpha_graph_map = check_alpha_folders(GRAPH_DIR, OUTPUT_DIR, alphas) 
     
    lock = mp.Lock() 
    cpu_count = min(mp.cpu_count(), MAX_CPU_COUNT)
    print(f"Using {cpu_count} available CPUs for multiprocessing...")
    p = mp.Pool(cpu_count, initializer=init, initargs=(lock,))
    param_list = [] # add desired parameters here
    total_proc = 0
    for alpha in alphas:
        for r in rs:
            num_graphs = len(alpha_graph_map[alpha])
            total_trials = SIMS_PER_GRAPH * num_graphs
            print(f"Testing pair of parameters alpha={alpha}, r={r} at time {datetime.datetime.now()} with {total_trials} total trials over {num_graphs} graphs...")
	    
            for graph_dir in alpha_graph_map[alpha]:
                param_list.append((alpha, r, SIMS_PER_GRAPH, os.path.join(alpha_dir_map[alpha], graph_dir)))
                total_proc += 1

    p.starmap(run_sims, param_list)
    print()
    
    for i in range(total_proc):
        alpha, r, size_arr, vir_arr, graph_dir = OUT_Q.get()
        if (alpha, r) not in results:
            results[(alpha, r)] = defaultdict(list)
        results[(alpha, r)]["size"].append(size_arr)
        results[(alpha, r)]["virality"].append(vir_arr)
        # save intermediate result
        out_file = os.path.join(OUTPUT_DIR, stringify_alpha(alpha), os.path.basename(graph_dir), "sim_result.pkl")
        pkl.dump({"size": size_arr, "virality": vir_arr}, open(out_file, "wb"), -1)

    # merge results
    for alpha in alphas:
        for r in rs:
            results[(alpha, r)]["size"] = np.concatenate(results[(alpha, r)]["size"])
            results[(alpha, r)]["virality"] = np.concatenate(results[(alpha, r)]["virality"])

    save_results(results, os.path.join(OUTPUT_DIR, RESULTS_FILENAME))
