#!/bin/bash

#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=5
#SBATCH --mem=20G               # memory per node
#SBATCH --time=00:05:00
#SBATCH --job-name="FetchReachDense-v1"
#SBATCH --account=def-zaiane
#SBATCH --output=FetchReachDense-v1-test-%j.out
#SBATCH --mail-user=taghianj@ualberta.ca
#SBATCH --mail-type=ALL

parallel < /home/taghianj/scratch/SAC_GCN/scripts/ComputeCanada/cedar/FetchReachEnv-v0/test.txt