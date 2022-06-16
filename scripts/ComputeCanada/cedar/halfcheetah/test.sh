#!/bin/bash

#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=5
#SBATCH --mem=20G               # memory per node
#SBATCH --time=00:10:00
#SBATCH --job-name="HalfCheetah"
#SBATCH --account=def-zaiane
#SBATCH --output=HalfCheetah-test-%j.out
#SBATCH --mail-user=taghianj@ualberta.ca
#SBATCH --mail-type=ALL

parallel < /home/taghianj/scratch/SAC_GCN/scripts/ComputeCanada/cedar/halfcheetah/test.txt