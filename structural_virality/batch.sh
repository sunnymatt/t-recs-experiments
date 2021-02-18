#!/bin/bash
#SBATCH --job-name=sv-1m-200cpu
#
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=200
#SBATCH --mem-per-cpu=2G
#SBATCH --time=98:00:00

#
#SBATCH --mail-type=all
#SBATCH --mail-user=mdsun@princeton.edu
python run_sim.py
