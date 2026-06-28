#!/bin/bash

#SBATCH --time=03:59:59
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=40GB
#SBATCH --output="opt_output.log"
#SBATCH --error="opt_error.log"

set -euo pipefail

cd ~/thesis

source ~/thesis/.venv/bin/activate
export PYTHONPATH=src

mkdir -p results
mkdir -p data/generated

rm -f data/generated/ghosts.txt
rm -f data/generated/high_logppl_scores.csv
rm -rf data/generated/CT data/generated/MT data/generated/CNT data/generated/MNT

# remove old trained model before training
rm -rf ~/../../scratch/s5628237/trained-model

python -m data.optimize_pool
python -m training.train
python -m eval.mia | tee data/generated/high_logppl_mu3_3333.log

# preserve optimized result under a clear name
OPT_RUN=$(ls -td results/runs/* | head -1)

mkdir -p results/named
rm -rf results/named/high_logppl_mu3_3333
cp -r "$OPT_RUN" results/named/high_logppl_mu3_3333

cp data/generated/high_logppl_mu3_3333.log results/named/high_logppl_mu3_3333/eval.log || true
cp data/generated/high_logppl_scores.csv results/named/high_logppl_mu3_3333/high_logppl_scores.csv || true
cp data/generated/ghosts.txt results/named/high_logppl_mu3_3333/ghosts.txt || true
cp opt_output.log results/named/high_logppl_mu3_3333/slurm_output.log || true
cp opt_error.log results/named/high_logppl_mu3_3333/slurm_error.log || true
cp config.yaml results/named/high_logppl_mu3_3333/config.yaml || true