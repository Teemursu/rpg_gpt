#!/bin/bash
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:a100:1
#SBATCH -J gpt_rpg_gptj
#SBATCH -o logfiles/gptrpg_gptj.out.%j
#SBATCH -e logfiles/gptrpg_gptj.err.%j
#SBATCH --account=project_2001403
#SBATCH --cpus-per-task=32

export PATH=$PATH:/users/poyhnent/.local/bin

source /scratch/project_2001403/poyhnent/venv/bin/activate

module purge
module load pytorch/1.13

srun singlegpu_gptj.py
