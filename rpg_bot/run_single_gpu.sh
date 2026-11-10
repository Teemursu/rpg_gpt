#!/bin/bash
#SBATCH -n 1
#SBATCH -p gputest
#SBATCH -t 00:15:00
#SBATCH --mem=300G
#SBATCH --gres=gpu:v100:1
#SBATCH -J gpt_rpg
#SBATCH -o logfiles/gptrpg.out.%j
#SBATCH -e logfiles/gptrpg.err.%j
#SBATCH --account=project_2001403
##SBATCH --mail-type=ALL
##SBATCH --mail-user=your.email@example.com   # optional: uncomment and set your address

source /scratch/project_2001403/poyhnent/venv/bin/activate 

srun python3 train_dialogpt.py
