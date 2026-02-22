#!/bin/bash

# run_tailored_outliers.sh
# 
# Usage: ./scripts/run_tailored_outliers.sh <benchmark_results_directory>
# Example: ./scripts/run_tailored_outliers.sh results/run_2026-02-12-08-14-35

if [ -z "$1" ]; then
    echo "Error: Please provide the target threshold directory."
    echo "Usage: $0 <benchmark_results_dir>"
    exit 1
fi

RESULTS_DIR=$1
OUTLIER_SCRIPT="scripts/extract_outliers.py"
OUTPUT_FILE="outlier_targets.txt"
JULIA_SCRIPT="experiments/generate_outliers_dataset.jl"

echo "=== Outlier Triage Pipeline ==="
echo "Target Directory: $RESULTS_DIR"

# Step 1: Extract filenames via Python pipeline
echo "Phase 1: Computing +2SD Layers from JSON Records..."

python3 "$OUTLIER_SCRIPT" "$RESULTS_DIR" > "$OUTPUT_FILE"

# Check if there are any outliers found
NUM_OUTLIERS=$(wc -l < "$OUTPUT_FILE" | tr -d ' ')

if [ "$NUM_OUTLIERS" -eq 0 ]; then
    echo "Result: No statistical layer outliers found in the target directory."
    rm "$OUTPUT_FILE"
    exit 0
fi

echo "Result: Extracted $NUM_OUTLIERS mathematically defined outliers."
echo "Target List saved temporarily to $OUTPUT_FILE"
echo ""

# Step 2: Inject List into Julia Data Builder
echo "Phase 2: Bootstrapping 3-Qubit Tailored Rescues..."

# Extract the N_vars from the first file to pass to Julia (e.g. sat_10_vars...)
FIRST_FILE=$(head -n 1 "$OUTPUT_FILE")
N_VARS=$(echo "$FIRST_FILE" | grep -oE "sat_[0-9]+_vars" | grep -oE "[0-9]+")

if [ -z "$N_VARS" ]; then
    echo "Error: Could not determine N_vars from outlier filename format ($FIRST_FILE)."
    exit 1
fi

echo "Detected N=$N_VARS"
echo "Executing Julia backend..."

# Run the julia worker natively 
julia "$JULIA_SCRIPT" --n_vars "$N_VARS" --outliers_file "$OUTPUT_FILE"

# Cleanup
rm "$OUTPUT_FILE"
echo "=== Pipeline Completed Successfully ==="
