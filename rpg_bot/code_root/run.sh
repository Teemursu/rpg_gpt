#!/bin/bash
#SBATCH -n 80
#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH --mem=350G
#SBATCH --gres=gpu:v100:1,nvme:1000
#SBATCH -J gpt_rpg
#SBATCH -o logfiles/gptrpg.out.%j
#SBATCH -e logfiles/gptrpg.err.%j
#SBATCH --account=project_2001403
##SBATCH --mail-type=ALL
##SBATCH --mail-user=your.email@example.com   # optional: uncomment and set your address

source /scratch/2001403/poyhnent/venv/bin/activate 

python train_dialogpt.py
