#!/bin/bash

#SBATCH --time=03:59:59
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=40GB
#SBATCH --output="full_output.log"
#SBATCH --error="full_error.log"

set -euo pipefail

cd ~/thesis

rm data/generated/ghosts.txt

source ~/thesis/.venv/bin/activate

mkdir -p results

export PYTHONPATH=src

python -m training.train
python -m eval.mia | tee results/mia_eval_test.log