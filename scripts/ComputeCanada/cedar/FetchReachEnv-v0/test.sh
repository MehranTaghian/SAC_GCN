#!/bin/bash
#!/bin/bash
#SBATCH --account=def-zaiane
#SBATCH --gpus-per-node=1
#SBATCH --mem=20G               # memory per node
#SBATCH --time=00:05:00
#SBATCH --job-name="FetchReachEnv-v0"
#SBATCH --account=def-zaiane
#SBATCH --mail-user=taghianj@ualberta.ca
#SBATCH --mail-type=ALL

parallel < test.txt