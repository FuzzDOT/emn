#!/bin/bash
#SBATCH --job-name=emn_exp3
#SBATCH --output=logs/exp3_%j.out
#SBATCH --error=logs/exp3_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
# NCSU HPC — Henry2, 8× A100 for parallel MC Dropout inference
# Submit: sbatch scripts/run_exp3.slurm

set -euo pipefail

module load cuda/11.8
module load anaconda3

NGPU=${NGPU:-8}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

conda activate emn

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_HOME="${REPO_ROOT}/.hf_cache"
mkdir -p "${HF_HOME}" logs results figures

echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    ${SLURM_NODELIST}"
echo "GPUs:    ${NGPU}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Inference is single-process (model fits on 1 A100)
# Multi-GPU used for batched parallel method evaluation
torchrun \
  --nproc_per_node="${NGPU}" \
  experiments/exp3_confabulation.py \
  --model-name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --seed 42 \
  --mc-passes 20 \
  --device cuda:0 \
  --results-dir results \
  --figures-dir figures

echo "Experiment 3 complete. Results in results/"
