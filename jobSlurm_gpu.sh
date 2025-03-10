#!/bin/bash --login
#SBATCH --job-name IGM
#SBATCH --error IGM-%j.error
#SBATCH --output IGM-%j.out
#SBATCH --time 00-1:30:00
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 80G
#SBATCH --partition gpu
#SBATCH --gres=gpu:1

module load gcc/12.3.0
module load cuda/11.8.0
module load cudnn/8.7.0.84-11.8

conda activate dle


# python train_emulator.py
python invert.py
