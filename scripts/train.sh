#!/bin/bash

#SBATCH --time=01:59:59
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=40GB
#SBATCH --output="train_output.log"
#SBATCH --error="train_error.log"

set -euo pipefail

cd ~/thesis

rm -rf data/generated/*

source ~/thesis/.venv/bin/activate

export PYTHONPATH=src

python -m training.train
python -m eval.mia | tee results/training_$(date +%Y%m%d_%H%M%S).log