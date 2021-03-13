#!/bin/bash
#SBATCH --job-name=a_2-3_r_0-5_v9
#
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=6G
#SBATCH --time=10:00:00

#
#SBATCH --mail-type=all
#SBATCH --mail-user=mdsun@princeton.edu
srun python run_sim.py --alphas 2.3 --rs 0.5 --sims_per_graph 1000 --output_dir exps/a_2-3_r_0-5_v9
