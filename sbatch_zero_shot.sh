#!/bin/bash
#SBATCH -A class_cse57388366fall2025
#SBATCH -J gsm8k_bootstrap          # Job name
#SBATCH -N 1                        # Number of nodes
#SBATCH -c 16                        # Number of CPU cores
#SBATCH --gres=gpu:a100:1           # Request 2 A100 GPUs
#SBATCH -t 0-02:00:00               # Runtime (d-hh:mm:ss)
#SBATCH -p general                  # Partition name (adjust if needed)
#SBATCH -q public                   # QoS
#SBATCH -o slurm.%j.out             # STDOUT file (%j = JobID)
#SBATCH -e slurm.%j.err             # STDERR file
#SBATCH --mail-type=ALL             # Notify on start, end, fail
#SBATCH --mail-user="%u@asu.edu"    # Replace %u with your actual ASU username
#SBATCH --export=NONE               # Purge environment for reproducibility

# -------------------- Environment Setup --------------------
module load mamba/latest
source activate jinal_env

# -------------------- Navigate to working directory --------------------
cd ~/Star

# -------------------- Run the Python script --------------------
#python bootstrapped_data_generation.py
python zero_shot.py 
