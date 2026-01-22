import Pkg
Pkg.activate(".")

import JSON
import Dates
import Random

# Include project modules
include(joinpath(@__DIR__, "..", "src", "MIS_TETRIS_ADAPT.jl"))
import .MIS_TETRIS_ADAPT: TetrisConfig, BenchmarkResult, BruteForceResult, TetrisResult
import .MIS_TETRIS_ADAPT: run_greedy_tetris, run_bruteforce, parse_cnf_file


"""
    generate_scaling_dataset(n_vars::Int, type::String, num_instances::Int; seed::Int=2000)

Generates a dataset for a specific N and type using the python script.
type: "balanced" or "triangle"
"""
function generate_scaling_dataset(n_vars::Int, type::String, num_instances::Int; seed::Int=2000)
    python_script = joinpath(@__DIR__, "..", "src", "dataset", "satqubolib_max3sat.py")
    venv_python = joinpath(@__DIR__, "..", "venv", "bin", "python3")

    ratio = 4.24

    cmd = `$venv_python $python_script $n_vars $ratio $num_instances --seed $seed --type $type`

    run(cmd)
end

"""
    run_scaling_benchmark()

Main entry point for the scaling benchmark.
"""
function run_scaling_benchmark()
    # 1. Configuration
    qubit_counts = [14, 15, 16]
    instances_per_type = 25
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
    output_file = joinpath(results_dir, "benchmark_scaling_$(timestamp).json")

    println("=== Starting Scaling Benchmark ===")
    println("N values: $qubit_counts")
    println("Instances per setting: $instances_per_type (Balanced) + $instances_per_type (Triangle)")
    println("Base Seed: $base_seed")
    println("Output: $output_file")

    # Consolidated results
    all_results = []

    # 2. Main Loop Over N
    for n_vars in qubit_counts
        println("\n>>> Processing N = $n_vars <<<")

        for type in ["balanced", "triangle"]
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
            target_files = filter(f -> endswith(f, ".cnf"), all_files)

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

                # A. Run Brute Force
                # Calculate timing
                t_start_bf = time()
                bf_result = run_bruteforce(instance)
                t_bf = time() - t_start_bf

                opt_energy = bf_result.approx_hamiltonian_energy

                # B. Run Greedy Tetris
                config = TetrisConfig(
                    adapt_type="greedy",
                    initial_gamma=0.01,
                    hamiltonian_type="approximate",
                    num_shots=1000
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
                    "seed" => current_seed + idx - 1,

                    # Brute Force Stats
                    "time_bf" => t_bf,
                    "energy_bf" => opt_energy,
                    "satisfaction_bf" => bf_result.best_satisfaction_count,
                    "satisfaction_bf_percent" => bf_result.percent_satisfied_clauses,

                    # Tetris Stats
                    "time_tetris" => tetris_result.total_runtime,
                    "energy_tetris" => tetris_result.final_energy,
                    "iterations" => tetris_result.num_iterations,
                    "layers" => tetris_result.num_adapt_layers,
                    "success" => tetris_result.success,
                    "hamiltonian_terms" => tetris_result.hamiltonian_terms,
                    "satisfaction_tetris_percent" => tetris_result.percent_satisfied_clauses
                )

                push!(all_results, res_entry)

                # Print progress every 10
                if idx % 10 == 0 || idx == length(experiment_files)
                    print("\r    Processed $idx/$(length(experiment_files))")
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
            JSON.print(f, all_results, 2)
        end
        println("  -> Saved intermediate results for N=$n_vars")
    end

    println("\n=== Scaling Benchmark Complete ===")
    println("Results saved to $output_file")
end

run_scaling_benchmark()
