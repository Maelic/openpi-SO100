#! /bin/bash
#SBATCH -A hpc2n2025-125
#SBATCH --time=10:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-node=h100:1
#SBATCH --error=/proj/nobackup/hpc2n2025-125/openpi-SO100/jobs/slurm-%j.err
#SBATCH --output=/proj/nobackup/hpc2n2025-125/openpi-SO100/jobs/slurm-%j.out

module pruge > /dev/null 2>&1

ml GCC/12.3.0 Python/3.11.3 CUDA/12.6.0 Rust/1.70.0 OpenMPI/4.1.5

uv run scripts/compute_norm_stats.py --config-name pi0_so101_low_mem_finetune --repo-id "maelic/grasp_fork"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_so101_low_mem_finetune  --data.repo_id "maelic/grasp_fork" --num_train_steps 30_000 --exp-name=pi0_so101_grasp_fork --overwrite