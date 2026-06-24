#!/bin/bash

#SBATCH --time=07:59:59
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=40GB
#SBATCH --output="lossgrad_mu3_output.log"
#SBATCH --error="lossgrad_mu3_error.log"

set -euo pipefail

cd ~/thesis
source ~/thesis/.venv/bin/activate
export PYTHONPATH=src

RUN_NAME="lossgrad_mu3_ng300_md900"
RESULT_DIR="results/sweeps/optimized_vs_random_budget/${RUN_NAME}"
MODEL_DIR="${TMPDIR:-/scratch/$USER}/thesis_model_current"

rm -rf "$RESULT_DIR"
rm -rf data/generated
rm -rf "$MODEL_DIR"

mkdir -p "$RESULT_DIR"
mkdir -p data/generated
mkdir -p "$MODEL_DIR"

python - <<PY
import yaml
from pathlib import Path

cfg_path = Path("config.yaml")
cfg = yaml.safe_load(cfg_path.read_text())

cfg["ghosts"]["mu"] = 3
cfg["ghosts"]["num_ghosts"] = 300
cfg["ghosts"]["total_ghosts"] = 600
cfg["ghosts"]["length"] = 10

cfg["model"]["output_dir"] = "$MODEL_DIR"

cfg["optimization"]["n"] = 600
cfg["optimization"]["steps"] = 3
cfg["optimization"]["topk"] = 8
cfg["optimization"]["alpha"] = 0.05
cfg["optimization"]["eps"] = 1.0e-8

cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

cp config.yaml "$RESULT_DIR/config.yaml"

python -m compileall src 2>&1 | tee "$RESULT_DIR/compile.log"

python -m data.optimize 2>&1 | tee "$RESULT_DIR/optimize.log"

python -m training.train 2>&1 | tee "$RESULT_DIR/train.log"

python -m eval.mia 2>&1 | tee "$RESULT_DIR/eval.log"

touch "$RESULT_DIR/DONE"

rm -rf "$MODEL_DIR"