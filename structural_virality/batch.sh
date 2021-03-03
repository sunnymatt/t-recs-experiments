#!/bin/bash
#SBATCH --job-name=sv-1m_alphas_2_5_2_7-vTESET
#
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=2G
#SBATCH --time=48:00:00

#
#SBATCH --mail-type=all
#SBATCH --mail-user=mdsun@princeton.edu
python run_sim.py
