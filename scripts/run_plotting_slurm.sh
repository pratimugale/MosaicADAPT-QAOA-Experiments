#!/bin/bash
#SBATCH --job-name=qaoa_plot
#SBATCH --output=logs/plotting_%j.out
#SBATCH --error=logs/plotting_%j.err
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Usage: sbatch scripts/run_plotting_slurm.sh <run_dir>

RUN_DIR_ARG=$1

if [ -z "$RUN_DIR_ARG" ]; then
    echo "Usage: sbatch scripts/run_plotting_slurm.sh <run_dir>"
    exit 1
fi

# Use the directory from which the job was submitted as the project root
PROJECT_ROOT="$SLURM_SUBMIT_DIR"

# If path is already absolute, use it directly; otherwise prepend PROJECT_ROOT
[[ "$RUN_DIR_ARG" = /* ]] && RUN_DIR="$RUN_DIR_ARG" || RUN_DIR="${PROJECT_ROOT}/${RUN_DIR_ARG}"

echo "Targeting RUN_DIR: $RUN_DIR"
echo "Project Root: $PROJECT_ROOT"
echo "Job started on $(hostname) at $(date)"

# 1. Generate Convergence, Trap, and Operator Plots
echo ">>> Generating Plots..."
python "$PROJECT_ROOT/scripts/generate_benchmark_plots.py" "$RUN_DIR" "$RUN_DIR/plots"

echo "Plotting and Analysis completed at $(date)"
