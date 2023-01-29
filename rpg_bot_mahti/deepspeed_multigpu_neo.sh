#!/bin/bash
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:a100:4
#SBATCH -J gpt_rpg
#SBATCH -o logfiles/gptrpg.out.%j
#SBATCH -e logfiles/gptrpg.err.%j
#SBATCH --account=project_2001403

export PATH=$PATH:/users/poyhnent/.local/bin

source /scratch/project_2001403/poyhnent/venv/bin/activate

module purge
module load python-data/3.9
module load pytorch/1.13

srun singularity_wrapper exec pip install --user accelerate
srun singularity_wrapper exec deepspeed deepspeed_multigpu_neo.py
