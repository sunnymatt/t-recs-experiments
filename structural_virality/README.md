This folder contains code related to the experiments to replicate results from "The structural virality of online diffusion" by Goel et al. (2016).

## Replicating Chaney et al. (2018)
1. Install T-RECS and related dependencies found in the scripts in this folder.
2. Create a folder where you will store the premade graphs.
3. Generate the sparse graph representation by running: `python create_graphs.py --graph_dir [YOUR_GRAPH_FOLDER]`. This will create nested folders inside the graph directory corresponding to different values of alpha, the power law sequence parameter (e.g., `YOUR_GRAPH_FOLDER/alpha_2-1` corresponds to a folder containing all graphs created from the power law sequence with `alpha=2.1`). By default, this command generates 25 graphs per alpha, but you can increase it using the `--num_graphs_per_alpha` parameter.
4. Create an output directory where the outputs of the simulation will be stored.
5. Next, you will need to simulate cascades on these graphs. You can do so with the command `python run_sim.py --graph_dir YOUR_GRAPH_DIR --output_dir YOUR_OUTPUT_DIRECTORY`. The flags `--alphas` and `--rs` allow you to perform a subset of additional experiments for specific values of `alpha` and `r`. You can also You can use the optional flag `--sims_per_graph` to control how many cascades per unique graph will be run. For example, if you generated 25 graphs for each level of alpha and `--sims_per_graph` is set to `100`, then there will be 2,500 total simulations run for each combination of `alpha, r`.
6. 