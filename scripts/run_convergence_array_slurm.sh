#!/bin/bash
#SBATCH --job-name=qaoa_conv_array
#SBATCH --output=logs/qaoa_conv_%A_%a.out
#SBATCH --error=logs/qaoa_conv_%A_%a.err
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=1-100

# Usage: sbatch scripts/run_convergence_array_slurm.sh <dataset_final_dir> <run_dir>

# Use SLURM_SUBMIT_DIR to find the project root
PROJECT_ROOT="$SLURM_SUBMIT_DIR"

ARG1=$1
ARG2=$2

# If path is already absolute, use it directly; otherwise prepend PROJECT_ROOT
[[ "$ARG1" = /* ]] && DATASET_FINAL="$ARG1" || DATASET_FINAL="${PROJECT_ROOT}/${ARG1}"
[[ "$ARG2" = /* ]] && RUN_DIR="$ARG2" || RUN_DIR="${PROJECT_ROOT}/${ARG2}"

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch scripts/run_convergence_array_slurm.sh <dataset_final_dir_relative> <run_dir_relative>"
    exit 1
fi

# Load modules below

# Explicitly set single threading environment variables
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Get the list of CNF files and pick the one corresponding to this array index
MAP_FILE=$(ls "$DATASET_FINAL"/*.cnf | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")

if [ -z "$MAP_FILE" ]; then
    echo "Error: No CNF file found for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Processing instance $SLURM_ARRAY_TASK_ID: $MAP_FILE"
echo "Job started on $(hostname) at $(date)"

# Run QAOA
echo ">>> Running QAOA for $(basename $MAP_FILE) (ID: $SLURM_ARRAY_TASK_ID)"
julia --project="$PROJECT_ROOT" "$PROJECT_ROOT/experiments/qaoa-sat.jl" "$MAP_FILE" "$RUN_DIR/qaoa" "$SLURM_ARRAY_TASK_ID"

echo "Job finished at $(date)"
