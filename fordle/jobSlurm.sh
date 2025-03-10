#!/bin/bash --login

#SBATCH --job-name IGM
#SBATCH --error IGM-%j.error
#SBATCH --output IGM-%j.out
#SBATCH -N 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 60G
#SBATCH -n 1
#SBATCH --partition cpu
#SBATCH --time 00:50:00

module load gcc miniconda3
conda activate igm

# python netcdfStats.py
python update_obs.py

