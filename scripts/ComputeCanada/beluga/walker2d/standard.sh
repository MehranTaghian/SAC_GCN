#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:4
#SBATCH --ntasks-per-node=40
#SBATCH --exclusive
#SBATCH --mem=191000M
#SBATCH --time=7-00:00
#SBATCH --job-name="Walker2d-15Seeds"
#SBATCH --account=def-zaiane
#SBATCH --output=Walker2d-%j.out
#SBATCH --mail-user=taghianj@ualberta.ca
#SBATCH --mail-type=ALL

parallel < /home/taghianj/scratch/SAC_GCN/scripts/ComputeCanada/run/walker2d/standard.txt
