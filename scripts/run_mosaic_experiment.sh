#!/bin/bash

# Orchestration Script for Convergence Analysis
# Goal: Run QAOA variants on a pre-generated benchmark dataset.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JULIA_NUM_THREADS=1

# 1. Handle Arguments
DATASET_DIR=${1:-"$PROJECT_ROOT/dataset/benchmark_final"}
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
RUN_NAME="benchmark_qaoa_variants_local_${TIMESTAMP}"
RUN_DIR="$PROJECT_ROOT/results/$RUN_NAME"

echo "=== Starting QAOA Benchmark Run ==="
echo "Dataset Directory: $DATASET_DIR"
echo "Results Directory: $RUN_DIR"

if [ ! -d "$DATASET_DIR" ] || [ -z "$(ls -A "$DATASET_DIR"/*.cnf 2>/dev/null)" ]; then
    echo "[ERROR] No dataset found in $DATASET_DIR. Please run the dataset generation script first."
    exit 1
fi

# 2. Setup Directories
mkdir -p "$RUN_DIR/qaoa" "$RUN_DIR/plots"
mkdir -p "$PROJECT_ROOT/logs"

# 3. Optimization Run (Parallel - 4 processes)
echo "=== [1/3] Running QAOA optimization on all instances (4 parallel processes) ==="
export PROJECT_ROOT RUN_DIR

ls "$DATASET_DIR"/*.cnf | awk '{print NR, $0}' | xargs -n 2 -P 4 bash -c '
    ID=$1
    MAP_FILE=$2
    echo ">>> Processing Instance $ID: $(basename "$MAP_FILE")"
    julia --project="$PROJECT_ROOT" "$PROJECT_ROOT/experiments/qaoa-sat.jl" "$MAP_FILE" "$RUN_DIR/qaoa" "$ID"
' _

echo "=== [SUCCESS] Optimization complete! ==="

echo "=== [2/3] Generating Plots ==="
python "$PROJECT_ROOT/scripts/generate_benchmark_plots.py" "$RUN_DIR" "$RUN_DIR/plots"
echo "=== [SUCCESS] Plots generated! ==="

if [ -f "$PROJECT_ROOT/dataset/canonical_formula_map.json" ]; then
    echo "=== [3/3] Archiving Canonical Formula Map ==="
    cp "$PROJECT_ROOT/dataset/canonical_formula_map.json" "$RUN_DIR/"
fi

echo "=== Benchmark Run Complete! Results are in $RUN_DIR ==="
