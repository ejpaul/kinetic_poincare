#!/bin/sh
#SBATCH --account=apam       
#SBATCH --job-name=simsopt       # The job name
#SBATCH -N 1                  # Number of nodes 
#SBATCH --time=0-05:00            # Time limit in D-HH:MM
#SBATCH --mem=170gb        # Memory per cpu core
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4

source ~/.bashrc
module load gcc/10.2.0
module load openmpi/gcc/64/4.1.5a1
module load anaconda
conda activate simsopt_072523

export OMP_NUM_THREADS=4
srun --mpi=pmix_v3 python trapped_map.py
