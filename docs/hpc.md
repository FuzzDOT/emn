# HPC Guide — NCSU Henry2 Cluster

## Setup

```bash
# Load modules
module load cuda/11.8
module load anaconda3

# Create environment
conda env create -f environment.yml
conda activate emn

# Verify
python -c "import torch; print(torch.cuda.is_available())"
python -c "import emn; print(emn.__version__)"
```

## Submitting Jobs

```bash
# Single experiments
sbatch scripts/run_exp1.slurm   # 1 GPU, 4h
sbatch scripts/run_exp2.slurm   # 4 GPUs, 12h, DDP
sbatch scripts/run_exp3.slurm   # 8 GPUs, 8h

# All experiments
bash scripts/run_all_experiments.sh --device cuda:0
```

## GPU Allocation

| Experiment | GPUs | Memory | Time |
| --- | --- | --- | --- |
| Exp 1 (MemoryAgentBench) | 1× A100 | 32 GB | 4h |
| Exp 2 (CIFAR100 CL) | 4× A100 | 64 GB | 12h |
| Exp 3 (Confabulation) | 8× A100 | 128 GB | 8h |

## DDP for Experiment 2

Experiment 2 uses `torchrun` with `--nproc_per_node=4`:

```bash
export NGPU=4
sbatch scripts/run_exp2.slurm
```

To override GPU count:
```bash
NGPU=1 sbatch scripts/run_exp2.slurm
```

## Monitoring

```bash
# Watch job output
tail -f logs/exp2_<job_id>.out

# Check GPU utilization
squeue -u $USER
sacct -j <job_id> --format=JobID,Elapsed,MaxRSS,State
```

## Fast Test (before full submission)

Always run with `--fast-test` first to verify the pipeline works:

```bash
bash scripts/run_all_experiments.sh --fast-test --device cuda:0
```

This runs 20 items per benchmark, 1 seed, 1 epoch — completes in ~5 minutes.
