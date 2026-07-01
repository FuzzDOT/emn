#!/bin/bash
#SBATCH --job-name=emn_exp2
#SBATCH --output=logs/exp2_%j.out
#SBATCH --error=logs/exp2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
# NCSU HPC — Henry2, 4× A100 80GB for DDP training
# Submit: sbatch scripts/run_exp2.slurm

set -euo pipefail

module load cuda/11.8
module load anaconda3

NGPU=${NGPU:-4}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

conda activate emn

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export MASTER_ADDR=localhost
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

mkdir -p logs results figures data/cifar100

echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    ${SLURM_NODELIST}"
echo "GPUs:    ${NGPU}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# DDP launch via torchrun
torchrun \
  --nproc_per_node="${NGPU}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  experiments/exp2_continual_learning.py \
  --seeds 42 43 44 \
  --epochs 5 \
  --device cuda \
  --results-dir results \
  --figures-dir figures

echo "Experiment 2 complete. Results in results/"
