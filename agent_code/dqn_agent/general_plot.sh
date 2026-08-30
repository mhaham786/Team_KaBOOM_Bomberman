#!/bin/bash

#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --mem=8GB  # RAM
#SBATCH --gres=gpu:1  # number of GPUs
#SBATCH --ntasks=1  # 
#SBATCH --cpus-per-task=2  # num of cpus
#SBATCH --output=%j.out  # regular print outputs
#SBATCH --error=%j.err   # error messages
#SBATCH --job-name=general_plot_test  # my job name

module load devel/miniconda  

conda activate python_env  # activate my custom environment

python main.py play --no-gui --agents dqn_agent  --train 1 --scenario coin-heaven --n-rounds 5000  # run python script

nvidia-smi  # show gpu usage

