#!/bin/bash

#SBATCH --time=00:60:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem=40GB
#SBATCH --output="output.log"
#SBATCH --error="error.log"

cd ~/thesis

source ~/thesis/.venv/bin/activate

cd src

python test.py