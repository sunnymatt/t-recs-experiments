# Matthew Sun, 2/2/2021
# Running structural virality simulations from premade graphs in a particular folder

from collections import defaultdict
import datetime
import multiprocessing as mp
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError
import os
import pprint
import pickle as pkl
import sys
from scipy.sparse import load_npz
import numpy as np
from trecs.models import BassModel
from create_graphs import stringify_alpha, stringify_r
from graph_utils import setup_experiment, popularity
from guppy import hpy
import errno
import argparse


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


def run_sims(alpha, r, proc_id, sims_per_graph, graph_dir, output_dir):
    """ Runs set of simulations on a set of premade graphs
        for a given level of alpha and r on one particular
        graph. A total of sims_per_graph simulations
        will be run. It then places the resulting
        sizes / SV values as a dictionary in the out_q Queue.
    """
    # where results will eventually be stored
    out_dir = os.path.join(output_dir, stringify_alpha(alpha), stringify_r(r), os.path.basename(graph_dir))
    out_file = os.path.join(out_dir, f"sim_result_{proc_id}.pkl")
    graph_sim_log_file = open(os.path.join(out_dir, f"log_{proc_id}.txt"), "a+")
    # redirect ouptut as desired
    err_file = open(os.path.join(out_dir, f"output_{proc_id}.txt"), "a+")
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
        if j % 10 == 0:
            # debug print heap size 
            h = hpy()
            print_to_log(f"Memory usage: {h.heap().size / 1000000}", graph_sim_log_file)
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

def merge_results(results, output_dir):
    """
    Take outputs from individual experiments and merge into one giant results file
    """
    for (root, dirs, files) in os.walk(output_dir, topdown=True):
        for f in files:
            if "sim_result" in f: # file stores results
                try:
                    result = pkl.load(open(os.path.join(root, f), "rb"))
                    alpha, r = result["alpha"], result["r"]
                    if (alpha, r) not in results:
                        results[(alpha, r)] = {}
                        results[(alpha, r)]["size"] = list()
                        results[(alpha, r)]["virality"] = list()
                    results[(alpha, r)]["size"].append(result["size"])
                    results[(alpha, r)]["virality"].append(result["virality"])
                except:
                    continue

    # now actually aggregate everything
    for alpha_r in results.keys():
        results[alpha_r]["size"] = np.concatenate(results[alpha_r]["size"])
        results[alpha_r]["virality"] = np.concatenate(results[alpha_r]["virality"])

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='running simulations')
    parser.add_argument('--graph_dir', type=str, default='graphs_1m')
    parser.add_argument('--sims_per_graph', type=int, default=1000)
    parser.add_argument('--sims_per_proc', type=int, default=25)
    parser.add_argument('--results_filename', type=str, default='merged_results.pkl')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_cpu_count', type=int, default=40) 
    parser.add_argument('--alphas', type=float, nargs='+', default=[2.1, 2.3, 2.5, 2.7, 2.9]) 
    parser.add_argument('--rs', type=float, nargs='+', default=[0.1, 0.3, 0.5, 0.7])
    args = parser.parse_args()
    arg_dict = vars(args)

    results = {}
    try:
        os.makedirs(arg_dict["output_dir"])
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    output_path = arg_dict["output_dir"]
    print(f"Writing results to {output_path}...")
    # write global params to file
    with open(os.path.join(arg_dict["output_dir"], "args.txt"), "w") as log_file:
        pprint.pprint(arg_dict, log_file)
       
    alphas, rs = arg_dict["alphas"], arg_dict["rs"] 
    alpha_dir_map, alpha_graph_map = process_input_output_dirs(arg_dict["graph_dir"], arg_dict["output_dir"], alphas, rs) 
     
    cpu_count = min(mp.cpu_count(), arg_dict["max_cpu_count"])
    print(f"Using {cpu_count} available CPUs for multiprocessing...")
    param_dict = {} # map ID to list of parameters

    with ProcessPool(max_workers=cpu_count, max_tasks=1) as pool:
        # callback function executes when task has completed. if timeout,
        # retry the task
        total_tasks = len(alphas) * len(rs) * sum([len(v) for k, v in alpha_graph_map.items()])
        def task_done(task_id):
            def callback(future):
                try:
                    result = future.result()
                except TimeoutError as error:
                    print(f"Function took longer than {error.args[1]} seconds with params {param_dict[task_id]}...")
                except ProcessExpired as error:
                    print(f"Function raised {error} with params {param_dict[task_id]}")
                    print(f"Exit code: {error.exitcode}")
                except Exception as error:
                    print(f"Function raised {error}")
                    print(error.traceback)
            return callback
       
        # schedule the simulations to run 
        cur_id = 0
        for alpha in alphas:
            for r in rs:
                num_graphs = len(alpha_graph_map[alpha])
                total_trials = arg_dict["sims_per_graph"] * num_graphs
                print(f"Queueing pair of parameters alpha={alpha}, r={r} at time {datetime.datetime.now()} with {total_trials} total trials over {num_graphs} graphs...")
 
                for graph_dir in alpha_graph_map[alpha]:
                    num_jobs = int(arg_dict["sims_per_graph"] / arg_dict["sims_per_proc"])
                    for j in range(num_jobs):
                        param_dict[cur_id] = [alpha, r, j, arg_dict["sims_per_proc"], os.path.join(alpha_dir_map[alpha], graph_dir), arg_dict["output_dir"]]
                        future = pool.schedule(run_sims, args=param_dict[cur_id])
                        future.add_done_callback(task_done(cur_id))
                        cur_id += 1
        pool.close()
        pool.join()

    print("Merging simulation files and writing to output directory...")
    merge_results(results, arg_dict["output_dir"])
    out_file = os.path.join(arg_dict["output_dir"], arg_dict["results_filename"])
    pkl.dump(results, open(out_file, "wb"), -1) 
