from scipy.sparse import csr_matrix, save_npz
import pickle as pkl
import os

from graph_utils import calc_avg_degree, implied_beta, scale_free_graph

num_nodes = 1000000
graph_dir = "graphs_1m"

for i in range(75):
    param_dict = {}
    subdir = f"{graph_dir}/{i}"
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    G = scale_free_graph(num_nodes, alpha=2.3)
    print(f"Finished creating graph {i} at time {datetime.datetime.now()}...")
    user_rep = nx.convert_matrix.to_scipy_sparse_matrix(G) # convert to scipy adjacency matrix
    save_npz(f'{graph_dir}/{i}/sparse_matrix.npz', user_rep)
    param_dict["k"] = calc_avg_degree(G)
    param_dict["beta"] = implied_beta(k, r)
    param_dict["num_nodes"] = num_nodes
    f = open(f'{graph_dir}/{i}/param.pkl', 'wb')
    pkl.dump(param_dict, f, -1)
    f.close()
    break
    