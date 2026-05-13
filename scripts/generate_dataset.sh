#!/bin/bash

# Orchestration Script for Convergence Analysis
# Goal: 50 Balanced + 50 Random (Satisfiable N=vars) across 3 protocols

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JULIA_NUM_THREADS=1


# 1. Handle Arguments
NUM_VARS=${1:-10}
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
RUN_NAME="benchmark_qaoa_variants_${NUM_VARS}_vars_${TIMESTAMP}"
RUN_DIR="$PROJECT_ROOT/results/$RUN_NAME"

echo "=== Starting Benchmark Run: $RUN_NAME ==="

# 2. Setup Directories
DATASET_RAW="$PROJECT_ROOT/dataset/benchmark_raw"
DATASET_SAT="$PROJECT_ROOT/dataset/benchmark_sat"
DATASET_REJECTED="$PROJECT_ROOT/dataset/benchmark_rejected"
DATASET_FINAL="$PROJECT_ROOT/dataset/benchmark_final"

mkdir -p "$DATASET_RAW" "$DATASET_SAT" "$DATASET_REJECTED" "$DATASET_FINAL"
mkdir -p "$RUN_DIR/qaoa" "$RUN_DIR/plots"
mkdir -p "$PROJECT_ROOT/logs"

# --- Dataset Caching Logic ---
FINAL_COUNT=$(ls "$DATASET_FINAL" 2>/dev/null | grep "\.cnf$" | wc -l)

if [ "$FINAL_COUNT" -eq 100 ]; then
    echo "=== [CACHE] Found 100 final instances in $DATASET_FINAL. Reusing existing benchmark set. ==="
    echo "=== [TIP] To regenerate the dataset, run: rm -rf $DATASET_FINAL/* ==="
else
    RAW_COUNT=$(ls "$DATASET_RAW" 2>/dev/null | grep "\.cnf$" | wc -l)
    if [ "$RAW_COUNT" -ge 300 ]; then
        echo "=== [CACHE] Found raw instances in $DATASET_RAW. Skipping Generation. ==="
    else
        echo "=== [1/2] Generating Raw Problem Instances (N=$NUM_VARS) ==="
        # Cleanup temporary dataset folders to ensure fresh generation if cache fails
        rm -f "$DATASET_RAW/"*.cnf "$DATASET_SAT/"*.cnf "$DATASET_REJECTED/"*.cnf

        # Generate 1000 balanced and 300 random to ensure we find enough satisfiable ones
        python "$PROJECT_ROOT/scripts/generate_max3sat_problem_instances.py" "$NUM_VARS" 1000 --type balanced --seed 2026
        python "$PROJECT_ROOT/scripts/generate_max3sat_problem_instances.py" "$NUM_VARS" 300 --type random --seed 2026

        # Move them to our local raw folder for filtering
        mv "$PROJECT_ROOT/dataset/satqubolib/balancedsat/"*.cnf "$DATASET_RAW/" 2>/dev/null
        mv "$PROJECT_ROOT/dataset/satqubolib/randomsat/"*.cnf "$DATASET_RAW/" 2>/dev/null
    fi

    echo "=== [2/2] Filtering for 100% Satisfiable Instances (Gurobi) ==="
    # Clear previous filtering results but keep RAW
    rm -f "$DATASET_SAT/"*.cnf "$DATASET_REJECTED/"*.cnf "$DATASET_FINAL/"*.cnf
    
    julia "$PROJECT_ROOT/scripts/filter_satisfiable.jl" "$DATASET_RAW" "$DATASET_SAT" "$DATASET_REJECTED"

    # Select exactly 50 of each type
    BALANCED_COUNT=$(ls "$DATASET_SAT" | grep "balanced" | wc -l)
    RANDOM_COUNT=$(ls "$DATASET_SAT" | grep "random" | wc -l)

    if [ "$BALANCED_COUNT" -lt 50 ] || [ "$RANDOM_COUNT" -lt 50 ]; then
        echo "[ERROR] Not enough satisfiable instances found (Balanced: $BALANCED_COUNT, Random: $RANDOM_COUNT)."
        exit 1
    fi

    ls "$DATASET_SAT" | grep "balanced" | head -n 50 | xargs -I {} cp "$DATASET_SAT/{}" "$DATASET_FINAL/"
    ls "$DATASET_SAT" | grep "random" | head -n 50 | xargs -I {} cp "$DATASET_SAT/{}" "$DATASET_FINAL/"

    FINAL_COUNT=$(ls "$DATASET_FINAL" | wc -l)
    echo "[SUCCESS] Final benchmark set verified: $FINAL_COUNT satisfiable instances."
fi
