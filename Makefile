# Variables
VENV_DIR = venv
SEED ?= 2000
INSTANCES ?= 50

# Main entry point
install:
	chmod +x install.sh
	./install.sh

install-python-deps:
	python3 -m venv $(VENV_DIR)
	./$(VENV_DIR)/bin/pip install -r requirements.txt

generate-dataset: install-python-deps
	./venv/bin/python3 src/dataset/satqubolib_max3sat.py 10 4.24 1 --seed 126 --type balanced

test-determinism: generate-dataset
	julia --project=. experiments/test_determinism.jl

# Run dataset experiment
experiment:
	julia --project=. experiments/run_greedy_dataset.jl

# Run benchmark (Greedy with Floor Stopper)
benchmark-floor:
	julia --project=. experiments/benchmark_floor_stopping.jl $(SEED)

# Scaling benchmark (Time vs Qubits)
benchmark-scaling:
	julia --project=. --threads=auto experiments/benchmark_scaling.jl $(SEED) $(INSTANCES)

# Plot scaling results (finds latest json)
plot-scaling:
	./venv/bin/python3 experiments/analysis/plot_scaling.py $$(ls -t results/benchmark_scaling_*.json | head -n 1) --output experiments/analysis/plots

# Run unit tests
test:
	julia --project=. test/runtests.jl

# Run RC2 solver test
test-rc2:
	./venv/bin/python3 src/exact_solvers/test_rc2.py

# Scaling benchmark (Gurobi Floor Stopper)
benchmark-gurobi-floor:
	julia --project=. experiments/benchmark_scaling_gurobi_floor.jl $(SEED) $(INSTANCES)

# Scaling benchmark (Exact Hamiltonian)
benchmark-scaling-exact-hamiltonian:
	julia --project=. experiments/benchmark_scaling_exact_hamiltonian.jl $(SEED) $(INSTANCES)

clear-dataset:
	rm -rf dataset/satqubolib/

.PHONY: dp install update-tetris install-python-deps init-julia-project generate-dataset test-determinism experiment benchmark-floor benchmark-scaling benchmark-gurobi-floor plot-scaling test test-rc2
