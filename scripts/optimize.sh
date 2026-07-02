#!/bin/bash

#SBATCH --time=01:59:59
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=40GB
#SBATCH --output="train_output.log"
#SBATCH --error="train_error.log"

export HF_HOME=/scratch/s5628237/huggingface
export HF_HUB_CACHE=/scratch/s5628237/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/s5628237/huggingface/transformers
export HF_DATASETS_CACHE=/scratch/s5628237/huggingface/datasets

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

set -euo pipefail

cd ~/thesis

rm -rf data/generated/*
rm -rf /scratch/s5628237/trained-model

source ~/thesis/.venv/bin/activate

export PYTHONPATH=src

python -m data.optimize_learnability
python -m training.train
python -m eval.mia | tee results/first_loss_drop_run.log