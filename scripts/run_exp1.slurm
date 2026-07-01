#!/bin/bash
#SBATCH --job-name=emn_exp1
#SBATCH --output=logs/exp1_%j.out
#SBATCH --error=logs/exp1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
# NCSU HPC — Henry2 cluster, A100 80GB
# Submit: sbatch scripts/run_exp1.slurm

set -euo pipefail

module load cuda/11.8
module load anaconda3

NGPU=${NGPU:-1}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

conda activate emn

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

mkdir -p logs results figures

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURM_NODELIST}"
echo "GPU:    ${CUDA_VISIBLE_DEVICES:-all}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python experiments/exp1_selective_forgetting.py \
  --seeds 42 43 44 \
  --device cuda:0 \
  --results-dir results \
  --figures-dir figures

echo "Experiment 1 complete. Results in results/"
