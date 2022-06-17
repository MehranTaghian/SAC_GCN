#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100s:4
#SBATCH --ntasks-per-node=40
#SBATCH --exclusive
#SBATCH --mem=185G
#SBATCH --time=15-00:00
#SBATCH --job-name="HalfCheetah-15Seeds"
#SBATCH --account=def-zaiane
#SBATCH --output=HalfCheetah-%j.out
#SBATCH --mail-user=taghianj@ualberta.ca
#SBATCH --mail-type=ALL

parallel < /home/taghianj/scratch/SAC_GCN/scripts/ComputeCanada/cedar/halfcheetah/standard.txt
