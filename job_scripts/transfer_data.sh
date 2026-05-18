#!/bin/bash
#SBATCH --job-name="transfer_data"
#SBATCH --time=24:00:00
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END
#SBATCH --output=/home/nfs/zli33/slurm_outputs/vlm_social/slurm_%j.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/vlm_social/slurm_%j.err

cp -r /tudelft.net/staff-umbrella/neon/zonghuan /tudelft.net/staff-umbrella/neon/GesBench