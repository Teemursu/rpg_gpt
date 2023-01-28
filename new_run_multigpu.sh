#!/bin/bash
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:a100:4
#SBATCH -J gpt_rpg
#SBATCH -o logfiles/gptrpg.out.%j
#SBATCH -e logfiles/gptrpg.err.%j
#SBATCH --account=project_2001403

source /scratch/project_2001403/poyhnent/venv/bin/activate 

module purge
module load pytorch/1.13

srun singularity_wrapper exec deepspeed new_script_multigpu.py --deepspeed --deepspeed_config my_ds_config.json 

