#!/bin/bash

#SBATCH --time=23:59:59
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=40GB
#SBATCH --output="sweep_optimization_budget_output.log"
#SBATCH --error="sweep_optimization_budget_error.log"

set -euo pipefail

cd ~/thesis
source ~/thesis/.venv/bin/activate

export PYTHONPATH=src

mkdir -p results/sweeps/optimized_vs_random_budget

python -m sweep_opt