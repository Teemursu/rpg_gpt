#!/bin/bash
#SBATCH --partition=dev-g
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 -p small-g --gpus-per-node 8 --mem=0 -t 01:00:00
#SBATCH -A project_462000007
#SBATCH --output=logs/slurm-%x-%j.out

date
rocm-smi

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
unset MIOPEN_DISABLE_CACHE

# debugging (noisy)
#export NCCL_DEBUG=INFO
#export RCCL_KERNEL_COLL_TRACE_ENABLE=1 
#export NCCL_DEBUG_SUBSYS=INIT,COLL

#No idea what these do, might work without them
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1

#Set HF cache to reside on scratch (a good idea, please change to your own)
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/scratch/project_462000007/hatanpav/output/datasets/
export HF_METRICS_CACHE=/scratch/project_462000007/hatanpav/output/metrics/
export HF_MODULES_CACHE=/scratch/project_462000007/hatanpav/output/modules/

#Load unstable pytorch module for the newest features and the most unstability, take not that unstable module is subject to change at any time
module use /appl/local/csc/soft/ai/modulefiles/
ml pytorch/unstable && echo "using pytorch/unstable"

#Without this something breaks. Can be an issue with my versions so you can safely remove this if yours works without this
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export MAIN_PY="/projappl/project_462000007/hatanpav/debugging/hf_hsa_error/test_scripts/multigpu_gptj.py"

set -x

srun singularity_wrapper exec python3 -m torch.distributed.run \
    --nproc_per_node=8 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(hostname):29420 $MAIN_PY

date