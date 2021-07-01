This folder contains code related to the experiments to replicate results from "How Algorithmic Confounding in Recommendation Systems Increases Homogeneity and Decreases Utility" by Chaney et al. (2018).

It also contains the code related to the content creator experiments in "T-RECS: A Simulation Tool to Study the Societal Impact of Recommender Systems" by Lucherini et al. (2021).

Both of these experiments make use of many of the assumptions of Chaney et al.'s studies, so they are placed here in the same folder. 

## Replicating Chaney et al. (2018)
1. Install T-RECS and all related dependencies needed for `creator_sim.py` to run.
2. Create a folder that will store the output of the simulation script.
3. Run the following command to perform the single-training simulations: `python replication_sim.py --output_dir [YOUR_OUTPUT_FOLDER]  --startup_iters 50 --sim_iters 50 --num_sims 400 --single_training`. 
4. Run the following command to perform the repeated training simulations: `python replication_sim.py --output_dir [YOUR_OUTPUT_FOLDER]  --startup_iters 10 --sim_iters 90 --num_sims 400 --repeated_training`.
5. Open the `chaney_replication.ipynb` notebook and run through the cells. You should reach a point where you'll need to replace a path (or set of paths) with your output directory.

## Content creator experiments (Lucherini et al. 2021)
1. Install T-RECS and all related dependencies needed for `replication_sim.py` to run.
2. Create a folder that will store the output of the simulation script.
3. Run the following command to perform the content creator simulations: `python creator_sim.py --output_dir [YOUR_OUTPUT_FOLDER] --startup_iters 10 --sim_iters 490 --repeated_training --metrics creator_item_homo creator_profiles --num_creators 5 --new_items_per_iter 5 --num_sims 400`.
4. Open the `content_creators.ipynb` notebook and run through the cells. You should reach a point where you'll need to replace a path (or set of paths) with your output directory.

**N.B.**: In the above instructions, we've set the `--num_sims` flag to `400`. However, in our experiments, we found it more practical to parallelize these operations on multiple nodes in our cluster; having 8 simulations running simultaneously allowed us to get results essentially overnight. (You'll see in the Jupyter notebooks that we have a function called `merge_results` to merge the results from multiple output folders.) Feel free to do the same parallelization, or simply use a smaller number of simulations (50 should be suitable to replicate the same trends).