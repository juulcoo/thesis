#!/bin/bash

#SBATCH --time=03:59:59
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=40GB
#SBATCH --output="output.log"
#SBATCH --error="error.log"

source ~/thesis/.venv/bin/activate

PYTHONPATH=src python -m training.train
PYTHONPATH=src python -m eval.mia