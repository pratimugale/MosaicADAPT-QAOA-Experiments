#!/bin/bash

# Slurm Script for Generating Max-3SAT Dataset and Running Benchmark
# Usage: sbatch generate_dataset_slurm.sh <N_VARS> [NUM_INSTANCES]
# Example: sbatch generate_dataset_slurm.sh 8 5

#SBATCH --job-name=tetris_benchmark
#SBATCH --output=tetris_benchmark_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=02:00:00
#SBATCH --mem=8G

# Capture arguments
N_VARS=${1:-10}        # Default to 10 if not provided
NUM_INSTANCES=${2:-50} # Default to 50 if not provided

CUR_DATE=$(date +'%Y-%m-%d_%H-%M-%S')
export OPENBLAS_NUM_THREADS=1
export JULIA_NUM_PRECOMPILE_TASKS=1
export JULIA_CPU_TARGET="generic"

# Ensure script directory (assuming scripts/ is one level deep)
# If running from root: ./scripts/generate_dataset_slurm.sh
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ROOT="$SCRIPT_DIR/.."
OUTPUT_DIR="$PROJECT_ROOT/results/$CUR_DATE"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$LOG_DIR"

echo "=== Benchmark Run Start: $CUR_DATE ==="
echo "N_VARS: $N_VARS"
echo "NUM_INSTANCES: $NUM_INSTANCES"
echo "Project Root: $PROJECT_ROOT"
echo "Output Dir: $OUTPUT_DIR"

# 1. Generate Dataset (Single Threaded Python)
echo ">>> Step 1: Generating Dataset..."

python3 "$PROJECT_ROOT/scripts/generate_max3sat_problem_instances.py" $N_VARS $NUM_INSTANCES --seed 42

if [ $? -ne 0 ]; then
    echo "Dataset generation failed!"
    exit 1
fi
echo "Dataset generated successfully."

# 2. Run Benchmark (Parallel Julia Workers)
N_WORKERS=5 # Adjust based on number of available cores
echo ">>> Step 2: Running Benchmark with $N_WORKERS workers..."

for ((i=1; i<=N_WORKERS; i++))
do
    echo "Starting Worker $i..."
    julia --project="$PROJECT_ROOT" "$PROJECT_ROOT/experiments/generate_dataset.jl" \
        --n_vars $N_VARS \
        --num_instances $NUM_INSTANCES \
        --worker_id $i \
        --n_workers $N_WORKERS \
        --output_dir "$OUTPUT_DIR" \
        --seed 42 \
        > "$LOG_DIR/worker_${i}.log" 2>&1 &
done

wait
echo "Benchmark completed. Results saved to $OUTPUT_DIR"
