
import Pkg
Pkg.activate(".")

import JSON
import Dates
import Random
using Printf

# Include project modules
include(joinpath(@__DIR__, "..", "src", "MIS_TETRIS_ADAPT.jl"))
import .MIS_TETRIS_ADAPT: TetrisConfig, BenchmarkResult, BruteForceResult, TetrisResult
import .MIS_TETRIS_ADAPT: run_greedy_tetris, run_bruteforce, parse_cnf_file
import ADAPT

# Include Gurobi Solver
include(joinpath(@__DIR__, "..", "src", "exact_solvers", "gurobi_solver.jl"))
using .GurobiSolver

"""
    generate_scaling_dataset(n_vars::Int, type::String, num_instances::Int; seed::Int=2000)

Generates a dataset for a specific N and type using the python script.
type: "balanced" or "triangle"
"""
function generate_scaling_dataset(n_vars::Int, type::String, num_instances::Int; seed::Int=2000)
    python_script = joinpath(@__DIR__, "..", "src", "dataset", "satqubolib_max3sat.py")
    venv_python = joinpath(@__DIR__, "..", "venv", "bin", "python3")

    base_ratio = 3.6
    
    # Seed the RNG for reproducibility of the ratio variations
    Random.seed!(seed)

    # We must loop to randomize ratio per instance
    for i in 0:(num_instances-1)
        # Randomize ratio +/- 20%
        # rand() gives [0, 1). We want [-0.2, 0.2].
        variation = (rand() * 0.4) - 0.2
        instance_ratio = base_ratio * (1.0 + variation)
        
        current_seed = seed + i
        
        # Call for SINGLE instance with specific ratio
        cmd = `$venv_python $python_script $n_vars $instance_ratio 1 --seed $current_seed --type $type`
        run(cmd)
    end
end

"""
    run_scaling_benchmark_gurobi_floor()

Main entry point for the scaling benchmark with Gurobi floor stopper.
"""
function run_scaling_benchmark_gurobi_floor()
    # 1. Configuration
    qubit_counts = [6, 8, 10]
    instances_per_type = 50
    base_seed = 2000

    # Usage: julia script.jl [SEED] [INSTANCES_PER_TYPE]
    if length(ARGS) >= 1
        base_seed = parse(Int, ARGS[1])
    end
    if length(ARGS) >= 2
        instances_per_type = parse(Int, ARGS[2])
    end

    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd-HH-MM-SS")
    results_dir = joinpath(@__DIR__, "..", "results")
    if !isdir(results_dir)
        mkdir(results_dir)
    end
    output_file = joinpath(results_dir, "benchmark_scaling_gurobi_floor_$(timestamp).json")

    println("=== Starting Scaling Benchmark (Gurobi Floor) ===")
    println("N values: $qubit_counts")
    println("Instances per setting: $instances_per_type (Balanced) + $instances_per_type (Triangle)")
    println("Base Seed: $base_seed")
    println("Using $(Base.Threads.nthreads()) threads for parallelization")

    println("Output: $output_file")
    flush(stdout)

    # Consolidated results container
    final_results = Dict(
        "timestamp" => timestamp,
        "threads" => Base.Threads.nthreads(),
        "config" => nothing,
        "results" => []
    )
    
    # Placeholder to capture config once
    captured_config = nothing

    # 2. Main Loop Over N
    for n_vars in qubit_counts
        println("\n>>> Processing N = $n_vars <<<")

        for type in ["balanced"] # ["balanced", "triangle"]
            println("\n  --- Generating $type dataset for N=$n_vars ---")

            current_seed = base_seed + (n_vars * 1000) + (type == "triangle" ? 500 : 0)
            generate_scaling_dataset(n_vars, type, instances_per_type; seed=current_seed)

            # Locate files
            if type == "balanced"
                dataset_dir = joinpath(@__DIR__, "..", "dataset", "satqubolib", "balancedsat")
            else
                dataset_dir = joinpath(@__DIR__, "..", "dataset", "satqubolib", "notrianglesat", "$(n_vars)nodes")
            end

            # Find the specific files we just generated
            all_files = readdir(dataset_dir)
            # Strict filtering: Must match seed AND N_vars to avoid cross-contamination
            target_files = filter(f -> endswith(f, ".cnf") && occursin("sat_$(n_vars)_vars", f), all_files)

            # Use specific seed-based filtering to be precise
            experiment_files = String[]
            for i in 0:(instances_per_type-1)
                s = current_seed + i
                # Filename pattern logic from python script: ...seed{seed}.cnf
                match = findfirst(f -> occursin("seed$s.cnf", f), target_files)
                if match !== nothing
                    push!(experiment_files, joinpath(dataset_dir, target_files[match]))
                end
            end

            println("  Found $(length(experiment_files)) files to process.")

            # Process Execution
            for (idx, cnf_path) in enumerate(experiment_files)
                # Parse
                instance = parse_cnf_file(cnf_path)
                instance["instance_id"] = idx
                formula = MIS_TETRIS_ADAPT.get_formula_as_struct(instance["formula"])

                # --- 1. Compute Gurobi Floor ---
                 # Get Approximate Hamiltonian (same as used by Tetris)
                H_approx = ADAPT.Hamiltonians.Max3SAT.get_approximate_hamiltonian(formula, n_vars)
                
                # Solve using Gurobi
                Q, offset = pauli_to_qubo(H_approx, n_vars)
                gurobi_energy, _ = solve_qubo_gurobi(Q, offset)
                
                # --- 2. Run Greedy Tetris ---
                config = TetrisConfig(
                    adapt_type="greedy",
                    initial_gamma=0.01,
                    hamiltonian_type="approximate",
                    num_shots=1000,
                    layer_stopper_max=n_vars*2,
                    energy_floor=gurobi_energy, # Set the floor!
                    floor_stopper_threshold=abs(0.01*gurobi_energy), # Stop if within 10% of floor
                    optimizer_tolerance=1e-3
                )

                t_start_tetris = time()
                tetris_result = run_greedy_tetris(config, instance)
                t_tetris = time() - t_start_tetris

                num_clauses = length(instance["formula"])

                pct_satisfied = 0.0
                if num_clauses > 0
                    pct_satisfied = tetris_result.sampled_expected_satisfaction / num_clauses
                end
                tetris_result.percent_satisfied_clauses = pct_satisfied

                # Record consolidated result
                res_entry = Dict(
                    "n_vars" => n_vars,
                    "type" => type,
                    "instance_idx" => idx,
                    "seed" => current_seed,

                    # Gurobi Baseline
                    "energy_floor_gurobi" => gurobi_energy,

                    # Tetris Stats
                    "time_tetris" => tetris_result.total_runtime,
                    "energy_tetris" => tetris_result.final_energy,
                    "energy_diff_vs_gurobi" => tetris_result.final_energy - gurobi_energy,
                    "iterations" => tetris_result.num_iterations,
                    "layers" => tetris_result.num_adapt_layers,
                    "success" => tetris_result.success,
                    "hamiltonian_terms" => tetris_result.hamiltonian_terms,
                    "satisfaction_tetris_percent" => tetris_result.percent_satisfied_clauses,
                    "stop_reason" => tetris_result.callback_flagged,
                    "adaptation_energies" => tetris_result.adaptation_energies
                )

                push!(final_results["results"], res_entry)
                
                # Capture config if not done
                if captured_config === nothing
                    captured_config = config
                    # Create descriptive config for the report
                    final_results["config"] = Dict(
                        "adapt_type" => config.adapt_type,
                        "initial_gamma" => config.initial_gamma,
                        "hamiltonian_type" => config.hamiltonian_type,
                        "num_shots" => config.num_shots,
                        "optimizer_tolerance" => config.optimizer_tolerance,
                        "optimizer_max_iterations" => config.optimizer_max_iterations,
                        
                        # Descriptive fields requested by user
                        "clause_randomness" => "+/- 20% of (3.6 * N)",
                        "floor_stopper_logic" => "1% of Gurobi Energy (abs(0.01 * E_gurobi))",
                        "energy_floor" => "Instance Specific (Gurobi)",
                        "layer_limit" => "2 * N"
                    )
                end

                # Print progress every 10
                if idx % 10 == 0 || idx == length(experiment_files)
                    println("    Processed $idx/$(length(experiment_files))")
                    flush(stdout)
                end
            end
            println("") # newline

            # Cleanup Files
            println("  Cleaning up generated files...")
            for f in experiment_files
                rm(f)
            end
        end

        # Intermediate Save (optional, but good practice for long runs)
        open(output_file, "w") do f
            JSON.print(f, final_results, 2)
        end
        println("  -> Saved intermediate results for N=$n_vars")
    end

    println("\n=== Scaling Benchmark (Gurobi Floor) Complete ===")
    println("Results saved to $output_file")
end

run_scaling_benchmark_gurobi_floor()
