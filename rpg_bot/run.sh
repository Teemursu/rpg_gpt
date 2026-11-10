#!/bin/bash
#SBATCH -n 1
#SBATCH -p gputest
#SBATCH -t 00:15:00
#SBATCH --mem=300G
#SBATCH --gres=gpu:a100:4
#SBATCH -J gpt_rpg
#SBATCH -o logfiles/gptrpg.out.%j
#SBATCH -e logfiles/gptrpg.err.%j
#SBATCH --account=project_2001403
##SBATCH --mail-type=ALL
##SBATCH --mail-user=your.email@example.com   # optional: uncomment and set your address

source /scratch/project_2001403/poyhnent/venv/bin/activate 

export RDZV_HOST=$(hostname)
export RDZV_PORT=29400

srun python3 -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
train_dialogpt_multiplegpu.py
