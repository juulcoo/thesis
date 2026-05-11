#!/bin/bash

#SBATCH --time=00:60:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem=40GB

cd ~/thesis

source ~/thesis/.venv/bin/activate

python train.py