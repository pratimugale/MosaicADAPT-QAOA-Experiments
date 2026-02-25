#!/bin/bash

# Slurm Script for Generating Max-3SAT Dataset and Running Benchmark
# Usage: sbatch generate_dataset_slurm.sh <N_VARS> [NUM_INSTANCES]
# Example: sbatch generate_dataset_slurm.sh 8 5

#SBATCH --job-name=tetris_sat_dataset_generation
#SBATCH --output=tetris_sat_dataset_generation_%j.out
#SBATCH --error=tetris_sat_dataset_generation_%j.err
#SBATCH --partition=idle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G

# Capture arguments
N_VARS=${1:-10}        # Default to 10 if not provided
NUM_INSTANCES=${2:-50} # Default to 50 if not provided
DATASET_NAME=${3:-"both"} # "balancedsat", "notrianglesat", or "both"

CUR_DATE=$(date +'%Y-%m-%d_%H-%M-%S')
export OPENBLAS_NUM_THREADS=1
export JULIA_NUM_PRECOMPILE_TASKS=1
export JULIA_CPU_TARGET="generic"

# Determine the project root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    # Running in Slurm
    # Add packages required on hpc
    vpkg_require python/3.13.1
    vpkg_require julia
    vpkg_require gcc/12.2.0
    
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

OUTPUT_DIR="$PROJECT_ROOT/results/$CUR_DATE"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$LOG_DIR"

echo "=== Benchmark Run Start: $CUR_DATE ==="
echo "N_VARS: $N_VARS"
echo "NUM_INSTANCES: $NUM_INSTANCES"
echo "Requested Dataset: $DATASET_NAME"
echo "Project Root: $PROJECT_ROOT"
echo "Output Dir: $OUTPUT_DIR"

# 1. Generate Dataset (Single Threaded Python)
echo ">>> Step 1: Checking/Generating Dataset..."

# Determine which types to generate
if [ "$DATASET_NAME" = "both" ]; then
    TYPES_TO_GEN="balanced notriangle"
elif [ "$DATASET_NAME" = "balancedsat" ]; then
    TYPES_TO_GEN="balanced"
elif [ "$DATASET_NAME" = "notrianglesat" ]; then
    TYPES_TO_GEN="triangle"
else
    echo "Unknown dataset: $DATASET_NAME"
    exit 1
fi

for TYPE in $TYPES_TO_GEN; do
    # Map back to directory names
    if [ "$TYPE" = "balanced" ]; then
        DIR_NAME="balancedsat"
        PY_TYPE="balanced"
    else
        DIR_NAME="notrianglesat"
        PY_TYPE="triangle"
    fi
    
    DATASET_DIR="$PROJECT_ROOT/dataset/satqubolib/$DIR_NAME"
    
    # Check if any .cnf files exist in the dataset directory for this N
    if [ -d "$DATASET_DIR" ] && [ "$(ls -A "$DATASET_DIR"/*_${N_VARS}_vars*.cnf 2>/dev/null)" ]; then
        echo "Dataset files for $TYPE (N=$N_VARS) found in $DATASET_DIR. Using existing."
    else
        echo "Generating new $TYPE dataset for N=$N_VARS..."
        python3 "$PROJECT_ROOT/scripts/generate_max3sat_problem_instances.py" $N_VARS $NUM_INSTANCES --seed 42 --type $PY_TYPE
        
        if [ $? -ne 0 ]; then
            echo "Dataset generation for $TYPE failed!"
            exit 1
        fi
        echo "$TYPE dataset generated successfully."
    fi
done

# 2. Run Benchmark (Parallel Julia Workers)
# Use $SLURM_CPUS_PER_TASK if available, otherwise default to 5
N_WORKERS=${SLURM_CPUS_PER_TASK:-5}
echo ">>> Step 2: Running Benchmark with $N_WORKERS workers..."

# On Slurm: wipe the ADAPT compiled cache so Julia rebuilds it with JULIA_CPU_TARGET=generic.
# This avoids "Unable to find compatible target" errors when nodes have different CPU features.
if [ -n "$SLURM_JOB_ID" ]; then
    echo ">>> Clearing stale precompile cache for all packages..."
    rm -rf ~/.julia/compiled/
fi

# Force Julia to use the user depot ONLY (prevents fallback to system-installed znver2 packages)
export JULIA_DEPOT_PATH="$HOME/.julia"

# Precompile once before spawning parallel workers (avoids race condition on first compile)
echo ">>> Precompiling project..."
julia --project="$PROJECT_ROOT" --cpu-target=generic -e 'using Pkg; Pkg.precompile()' \
    && echo ">>> Precompilation done." \
    || { echo "ERROR: Precompilation failed! Check logs."; exit 1; }

for ((i=1; i<=N_WORKERS; i++))
do
    echo "Starting Worker $i..."
    julia --project="$PROJECT_ROOT" --threads=1 --cpu-target=generic "$PROJECT_ROOT/experiments/generate_dataset.jl" \
        --n_vars $N_VARS \
        --num_instances $NUM_INSTANCES \
        --worker_id $i \
        --n_workers $N_WORKERS \
        --output_dir "$OUTPUT_DIR" \
        --seed 42 \
        --dataset_name "$DATASET_NAME" \
        > "$LOG_DIR/worker_${i}.log" 2>&1 &
done

wait
echo "Benchmark completed. Results saved to $OUTPUT_DIR"
