#!/bin/bash

set -euo pipefail

cd ~/thesis

source ~/thesis/.venv/bin/activate
export PYTHONPATH=src

mkdir -p results/figures

RUN_ID=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results/figures/${RUN_ID}"

mkdir -p "$OUT_DIR"

if [ "$#" -eq 0 ]; then
    echo "No run dirs given. Using all folders in results/named/"
    RUN_DIRS=(results/named/*)
else
    RUN_DIRS=("$@")
fi

echo "Plot output dir: $OUT_DIR"
echo "Using runs:"
for run in "${RUN_DIRS[@]}"; do
    echo "  $run"

    if [ ! -f "$run/metrics.csv" ]; then
        echo "Missing metrics.csv in $run"
        exit 1
    fi
done

python -m eval.plot_thesis \
    "${RUN_DIRS[@]}" \
    --out-dir "$OUT_DIR"

echo ""
echo "Done. Created:"
ls -lh "$OUT_DIR"