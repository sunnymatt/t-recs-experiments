# Matthew Sun, 2/2/2021
# Running structural virality simulations from premade graphs in a particular folder

from collections import defaultdict
import datetime
import multiprocessing as mp
import os
import pprint
import pickle as pkl
import sys
from scipy.sparse import load_npz
import numpy as np
from trecs.models import BassModel
from create_graphs import stringify_alpha, stringify_r
from graph_utils import setup_experiment, popularity

PARAMS = {
    "GRAPH_DIR": "graphs_1m",
    "SIMS_PER_GRAPH": 10,
    "RESULTS_FILENAME": "results.pkl",
    "OUTPUT_DIR": "exps/r_0-5_0-7_trials_TEST",
    "MAX_CPU_COUNT": 64,
    "ALPHAS": [2.1, 2.3, 2.5, 2.7, 2.9],
    "RS":  [0.5, 0.7],
    "NUM_RETRIES": 3,
}

# check folders that are supposed to exist actually do exist
# and create intermediate output folders
def process_input_output_dirs(graph_dir, output_dir, alphas, rs):
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
        for r in rs:
            for g in graph_subdirs:
                graph_outdir = os.path.join(output_alpha_dir, stringify_r(r), g)
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
    
def print_to_log(msg, file_obj):
    msg = f"{datetime.datetime.now().timestamp()}: {msg}"
    print(msg, file=file_obj, flush=True)

def run_sims(alpha, r, sims_per_graph, graph_dir):
    """ Runs set of simulations on a set of premade graphs
        for a given level of alpha and r on one particular
        graph. A total of sims_per_graph simulations
        will be run. It then places the resulting
        sizes / SV values as a dictionary in the out_q Queue.
    """
    # where results will eventually be stored
    out_dir = os.path.join(PARAMS["OUTPUT_DIR"], stringify_alpha(alpha), stringify_r(r), os.path.basename(graph_dir))
    out_file = os.path.join(out_dir, "sim_result.pkl")
    graph_sim_log_file = open(os.path.join(out_dir, "log.txt"), "a+")
    # redirect ouptut as desired
    err_file = open(os.path.join(out_dir, "output.txt"), "a+")
    sys.stdout = err_file
    sys.stderr = err_file
    if os.path.isfile(out_file):
        # simulation has already successfully occurred in a previous run!
        print_to_log(f"alpha={alpha}, r={r}: Already completed simulations on graph in {graph_dir} at time {datetime.datetime.now()}", graph_sim_log_file)
        graph_sim_log_file.close()
        return out_file
    # these will store results
    size_arr = np.zeros(sims_per_graph)
    vir_arr = np.zeros(sims_per_graph)
    # keep track of index of trial
    trial_idx = 0
    
    # find subfolders for each individual graph
    user_rep = load_npz(os.path.join(graph_dir, "sparse_matrix.npz"))
    # param_dict contains k, r, beta, and num_nodes
    param_dict = pkl.load(open(os.path.join(graph_dir, "param.pkl"), "rb")) 
    num_popular = 0 # keep track of size >= 100    
    print_to_log(f"alpha={alpha}, r={r}: Process pid = {os.getpid()} on graph in {graph_dir}", graph_sim_log_file)
    print_to_log(f"alpha={alpha}, r={r}: Starting {sims_per_graph} simulations on graph in {graph_dir} at time {datetime.datetime.now()} ...", graph_sim_log_file)
    for j in range(sims_per_graph):
        if j % 1 == 0:
            print_to_log(f"alpha={alpha}, r={r}: On iteration {j} of {sims_per_graph} on {graph_dir} at {datetime.datetime.now()}", graph_sim_log_file)
        simulation = setup_experiment(user_rep, param_dict["k"], r=r)
        simulation.run()
        size = popularity(simulation)
        size_arr[trial_idx] = size
        try:
            sv = simulation.get_structural_virality()
            vir_arr[trial_idx] = sv
            if size >= 100:
                num_popular += 1
                print_to_log(f"{num_popular} popular cascades on iteration {j} for alpha={alpha}, r={r} in {graph_dir}", graph_sim_log_file)
        except:
            vir_arr[trial_idx] = -1 # couldn't calculate virality
        trial_idx += 1
    # close open log file
    print_to_log(f"alpha={alpha}, r={r}: Completed {sims_per_graph} simulations on graph in {graph_dir} at time {datetime.datetime.now()}!", graph_sim_log_file)
    graph_sim_log_file.close()
    # example output folder: "sim_results/alpha_2-1/r_0-5/2/sim_results.pkl
    pkl.dump({"size": size_arr, "virality": vir_arr, "r": r, "alpha": alpha}, open(out_file, "wb"), -1)
    return out_file
   

def spawn_workers(num_cpus, param_list):
    """
    Creates multiprocessing pool to run simulations
    """
    with mp.Pool(num_cpus) as pool:
        result_files = pool.starmap(run_sims, param_list)
    return result_files
 
if __name__ == "__main__":
    results = {}
    # varying alpha and R
    if not os.path.exists(PARAMS["OUTPUT_DIR"]):
        os.makedirs(PARAMS["OUTPUT_DIR"])
    # write global params to file
    with open(os.path.join(PARAMS["OUTPUT_DIR"], "args.txt"), "w") as log_file:
        pprint.pprint(PARAMS, log_file)
       
    alphas, rs = PARAMS["ALPHAS"], PARAMS["RS"] 
    alpha_dir_map, alpha_graph_map = process_input_output_dirs(PARAMS["GRAPH_DIR"], PARAMS["OUTPUT_DIR"], alphas, rs) 
     
    cpu_count = min(mp.cpu_count(), PARAMS["MAX_CPU_COUNT"])
    print(f"Using {cpu_count} available CPUs for multiprocessing...")
    param_list = [] # add desired parameters here
    for alpha in alphas:
        for r in rs:
            num_graphs = len(alpha_graph_map[alpha])
            total_trials = PARAMS["SIMS_PER_GRAPH"] * num_graphs
            print(f"Queueing pair of parameters alpha={alpha}, r={r} at time {datetime.datetime.now()} with {total_trials} total trials over {num_graphs} graphs...")
	    
            for graph_dir in alpha_graph_map[alpha]:
                param_list.append((alpha, r, PARAMS["SIMS_PER_GRAPH"], os.path.join(alpha_dir_map[alpha], graph_dir)))
            
            # create empty results arrays
            results[(alpha, r)] = {"size": list(), "virality": list()}

    print("Starting multiprocessing pool...")
    result_files = spawn_workers(cpu_count, param_list)

    print()
    print('Iterating through results of simulations...') 
    
    for path in result_files:
        result = pkl.load(open(path, "rb"))
        alpha, r = result["alpha"], result["r"]
        results[(alpha, r)]["size"].append(result["size"])
        results[(alpha, r)]["virality"].append(result["virality"])

    for alpha_r in results.keys():
        results[alpha_r]["size"] = np.concatenate(results[alpha_r]["size"])
        results[alpha_r]["virality"] = np.concatenate(results[alpha_r]["virality"])
        
    save_results(results, os.path.join(PARAMS["OUTPUT_DIR"], PARAMS["RESULTS_FILENAME"]))
