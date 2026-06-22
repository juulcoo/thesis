#!/bin/bash

#SBATCH --time=03:59:59
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=40GB
#SBATCH --output="output.log"
#SBATCH --error="error.log"

set -euo pipefail

cd ~/thesis
source ~/thesis/.venv/bin/activate

mkdir -p results

export PYTHONPATH=src

python -m compileall src
python -m eval.mia | tee results/mia_eval_test.log