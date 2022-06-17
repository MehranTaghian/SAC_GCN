#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:4
#SBATCH --ntasks-per-node=40
#SBATCH --mem=1G
#SBATCH --time=00:03:00
#SBATCH --job-name="information"
#SBATCH --account=def-zaiane
#SBATCH --output=information-%j.out
#SBATCH --mail-user=taghianj@ualberta.ca
#SBATCH --mail-type=ALL

nvidia-smi