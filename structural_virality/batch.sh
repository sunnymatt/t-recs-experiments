#!/bin/bash
#SBATCH --job-name=25m_a_2-9_g_25
#
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=6G
#SBATCH --time=18:00:00

#
#SBATCH --mail-type=all
#SBATCH --mail-user=mdsun@princeton.edu
srun python parallel_test.py --graph_dir graphs_25m --alphas 2.9 --num_nodes 25000000 --graph_ids 25 
