#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100:4
#SBATCH --ntasks-per-node=24
#SBATCH --exclusive
#SBATCH --mem=125G
#SBATCH --time=6-00:00
#SBATCH --job-name="FetchReachEnv-v3"
#SBATCH --account=def-zaiane
#SBATCH --output=FetchReachEnv-v3-standard-%j.out
#SBATCH --mail-user=taghianj@ualberta.ca
#SBATCH --mail-type=ALL

parallel < /home/taghianj/scratch/SAC_GCN/scripts/ComputeCanada/cedar/FetchReachEnv-v3/standard.txt