#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=40GB
#SBATCH --output="output.log"
#SBATCH --error="error.log"

source ~/thesis/.venv/bin/activate

PYTHONPATH=src python -m eval.mia