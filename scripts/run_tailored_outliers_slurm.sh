#!/bin/bash

# run_tailored_outliers_slurm.sh
# 
# Usage: sbatch scripts/run_tailored_outliers_slurm.sh <benchmark_results_directory>
# Example: sbatch scripts/run_tailored_outliers_slurm.sh results/run_2026-02-12-08-14-35

#SBATCH --job-name=tetris_outliers_rescue
#SBATCH --output=tetris_outliers_rescue_%j.out
#SBATCH --error=tetris_outliers_rescue_%j.err
#SBATCH --partition=idle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=64G

if [ -z "$1" ]; then
    echo "Error: Please provide the target threshold directory."
    echo "Usage: sbatch $0 <benchmark_results_dir>"
    exit 1
fi

RESULTS_DIR=$1

export OPENBLAS_NUM_THREADS=1
export JULIA_NUM_PRECOMPILE_TASKS=1
export JULIA_CPU_TARGET="generic"

# Determine the project root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    # Running in Slurm
    # Add packages required on hpc
    vpkg_require python/3.13.1
    vpkg_require julia
    vpkg_require gcc/14.2
    
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
    
    # Check for venv, create if missing, install requirements
    VENV_DIR="$PROJECT_ROOT/venv"
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating venv at $VENV_DIR..."
        python3 -m venv "$VENV_DIR"
        source "$VENV_DIR/bin/activate"
        pip install -r "$PROJECT_ROOT/requirements.txt"
    else
        source "$VENV_DIR/bin/activate"
    fi
else
    # Running locally or directly, use the script's directory to find root
    SCRIPT_DIR="$(dirname "$(realpath "$0")")"
    PROJECT_ROOT="$SCRIPT_DIR/.."
    
    # Locally we assume venv exists or user handles it, but we can activate if found
    if [ -d "$PROJECT_ROOT/venv" ]; then
        source "$PROJECT_ROOT/venv/bin/activate"
    fi
fi

# Ensure working directory is project root
cd "$PROJECT_ROOT"

OUTLIER_SCRIPT="$PROJECT_ROOT/scripts/extract_outliers.py"
OUTPUT_FILE="$PROJECT_ROOT/outlier_targets_${SLURM_JOB_ID:-local}.txt"
JULIA_SCRIPT="$PROJECT_ROOT/experiments/generate_outliers_dataset.jl"

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
julia --project="$PROJECT_ROOT" "$JULIA_SCRIPT" --n_vars "$N_VARS" --outliers_file "$OUTPUT_FILE"

# Cleanup
rm "$OUTPUT_FILE"
echo "=== Pipeline Completed Successfully ==="
