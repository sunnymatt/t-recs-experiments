# Matthew Sun, 2/2/2021
# Running structural virality simulations from premade graphs
import os
import pickle as pkl
from scipy.sparse import load_npz
import numpy as np
from trecs.models import BassModel


GRAPH_DIR = "graphs_1m"
SIMS_PER_GRAPH = 100

# find subfolders for each individual graph
graph_subdirs = [f.name for f in os.scandir(GRAPH_DIR) if f.is_dir()]
print(f"Available graph directories: {', '.join(graph_subdirs)}")
print(f"Total graphs available: {len(graph_subdirs)}")

# varying alpha and R
alphas = [2.3]
rs = [0.1, 0.3, 0.5, 0.7, 0.9]

for alpha, r in zip(alphas, rs):
    print(f"Testing pair of parameters alpha={alpha}, r={r} at time {datetime.datetime.now()}...")
    
    total_trials = len(graph_subdirs) * SIMS_PER_GRAPH
    size_arr = np.zeros(total_trials)
    vir_arr = np.zeros(total_trials)
    trial_idx = 0
    
    for i, graph_subdir in enumerate(graph_subdirs):
        use_rep = load_npz(os.path.join(GRAPH_DIR, graph_subdir, "sparse_matrix.npz")) 
	# param_dict contains k, r, beta, and num_nodes
        param_dict = pkl.load(open(os.path.join(GRAPH_DIR, graph_subdir, "param.pkl"), "rb")) 
        
	if i % 10 == 0:
            print(f"\tOn graph {i} at time {datetime.datetime.now()}...")
            
        for j in range(trials_per_graph):
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

    results[(alpha, r)]["size"] = size_arr
    results[(alpha, r)]["virality"] = vir_arr
    
    print("")
