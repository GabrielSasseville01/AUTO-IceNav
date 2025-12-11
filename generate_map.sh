#!/bin/bash
#SBATCH -J IFT6162-IceNav-MapGen
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=long
#SBATCH --time=2:00:00

# Generate new map with given shape
python -m ship_ice_planner.experiments.generate_rand_exp