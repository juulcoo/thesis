#!/bin/bash

#SBATCH --time=03:59:59
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=40GB
#SBATCH --output="full_output.log"
#SBATCH --error="full_error.log"

set -euo pipefail

cd ~/thesis

rm -f data/generated/ghosts.txt
rm -f data/generated/high_logppl_scores.csv
rm -rf data/generated/CT data/generated/MT data/generated/CNT data/generated/MNT
rm -rf ~/../../scratch/s5628237/trained-model

source ~/thesis/.venv/bin/activate

mkdir -p results

export PYTHONPATH=src

# python -m data.optimize_pool
python -m training.train
# python -m eval.mia | tee data/generated/mia_high_logppl_mu3_3333.log
python -m eval.mia | tee data/generated/random_mu3_3333.log