#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:4
#SBATCH --ntasks-per-node=40
#SBATCH --exclusive
#SBATCH --mem=191000M
#SBATCH --time=14-00:00
#SBATCH --job-name="information"
#SBATCH --account=def-zaiane
#SBATCH --output=information-%j.out
#SBATCH --mail-user=taghianj@ualberta.ca
#SBATCH --mail-type=ALL

nvidia-smi