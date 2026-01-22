#!/bin/bash
#SBATCH --job-name=bench_scale      # Job name
#SBATCH --output=bench_%j.out  # Standard output log (files, not terminal)
#SBATCH --error=bench_%j.err   # Standard error log
#SBATCH --partition=idle            # Partition changed to 'idle'
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --cpus-per-task=8           # CPU cores
#SBATCH --mem=64G                   # Memory increased to 64G
#SBATCH --time=04:00:00             # Time limit 4 hours

vpkg_require python/3.13.1
vpkg_require julia

echo "Job started on $(hostname) at $(date)"

echo "Running make benchmark-scaling"

cd ../ && make benchmark-scaling

echo "Job finished at $(date)"
