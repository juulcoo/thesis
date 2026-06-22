#!/bin/bash

#SBATCH --time=23:59:59
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=40GB
#SBATCH --output="sweep_mu_output.log"
#SBATCH --error="sweep_mu_error.log"

set -euo pipefail

cd ~/thesis
source ~/thesis/.venv/bin/activate

export PYTHONPATH=src

python -m sweep_mu