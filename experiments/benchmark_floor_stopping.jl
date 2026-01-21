import Pkg
Pkg.activate(".")

import JSON
import Dates
import Random

# Include project modules
include(joinpath(@__DIR__, "..", "src", "MIS_TETRIS_ADAPT.jl"))
# Explicitly import needed symbols
import .MIS_TETRIS_ADAPT: TetrisConfig, TetrisResult, BruteForceResult, BenchmarkResult
import .MIS_TETRIS_ADAPT: run_greedy_tetris, run_bruteforce

"""
    generate_dataset_via_cli(num_instances::Int; seed::Int=123)
Generates a fresh dataset of 10-variable balanced Max-3-SAT instances using the Python script.
"""
function generate_dataset_via_cli(num_instances::Int; seed::Int=123)
    println("--- Generating Dataset ---")
    python_script = joinpath(@__DIR__, "..", "src", "dataset", "satqubolib_max3sat.py")
    venv_python = joinpath(@__DIR__, "..", "venv", "bin", "python3")

    # Hardcoded to 10 variables, 4.24 ratio as requested/benchmark standard
    cmd = `$venv_python $python_script 10 4.24 $num_instances --seed $seed --type balanced`

    println("Running command: $cmd")
    run(cmd)
    println("Dataset generation complete.")
end

"""
    run_benchmark_experiment()
Main entry point: generates dataset, runs Brute Force + Greedy Benchmark, saves results.
"""
function run_benchmark_experiment()
    # 1. Configuration
    dataset_seed = 2000 # Default
    if !isempty(ARGS)
        parsed_seed = tryparse(Int, ARGS[1])
        if parsed_seed !== nothing
            dataset_seed = parsed_seed
        else
            println("Warning: Could not parse argument '$(ARGS[1])' as integer seed. Using default $dataset_seed.")
        end
    end

    num_instances = 5   # Small batch for benchmark demo, user can increase

    println("=== Starting Benchmark: Brute Force + Greedy Tetris ===")
    println("Instances: $num_instances (Seed: $dataset_seed)")

    # 2. Generate Dataset
    generate_dataset_via_cli(num_instances, seed=dataset_seed)

    # 3. Locate files
    dataset_dir = joinpath(@__DIR__, "..", "dataset", "satqubolib", "balancedsat")
    all_files = readdir(dataset_dir)
    target_files = filter(f -> endswith(f, ".cnf") && occursin("10_vars", f), all_files)

    # Filter by specific seeds
    expected_seeds = [dataset_seed + i for i in 0:(num_instances-1)]
    experiment_files = String[]
    for seed in expected_seeds
        match = findfirst(f -> occursin("seed$seed.cnf", f), target_files)
        if match !== nothing
            push!(experiment_files, joinpath(dataset_dir, target_files[match]))
        else
            println("Warning: Could not find generated file for seed $seed")
        end
    end

    if isempty(experiment_files)
        error("No matching experiment files found.")
    end

    # 4. Run Loop
    benchmark_results = BenchmarkResult[]

    for (i, cnf_path) in enumerate(experiment_files)
        instance_filename = basename(cnf_path)
        println("\n--- Processing $i/$(length(experiment_files)): $instance_filename ---")

        # Use qualified access
        instance = MIS_TETRIS_ADAPT.parse_cnf_file(cnf_path)
        instance["instance_id"] = i

        # A. Run Brute Force
        println("[Brute Force] Finding exact solution...")
        bf_result = run_bruteforce(instance)

        opt_energy = bf_result.approx_hamiltonian_energy
        println("  -> Optimal Energy (Approx H): $opt_energy")
        println("  -> Max Clauses Satisfied: $(bf_result.best_satisfaction_count)")
        println("  -> Time: $(round(bf_result.execution_time, digits=3))s")

        # B. Run Greedy Tetris with Energy Floor
        println("[Greedy Tetris] Running with Energy Floor = $opt_energy...")

        # Configure Tetris with the found floor
        config = TetrisConfig(
            adapt_type="greedy",
            initial_gamma=0.01,
            hamiltonian_type="approximate",
            num_shots=1000,
            energy_floor=opt_energy
        )

        tetris_result = run_greedy_tetris(config, instance)

        # Populate Stats match run_greedy_dataset logic
        num_clauses = length(instance["formula"])
        if num_clauses > 0
            tetris_result.percent_satisfied_clauses = tetris_result.sampled_expected_satisfaction / num_clauses
        end
        tetris_result.num_clauses = num_clauses

        println("  -> Iterations: $(tetris_result.num_iterations)")
        println("  -> Layers: $(tetris_result.num_adapt_layers)")
        println("  -> Final Energy: $(tetris_result.final_energy) (Target: <= $opt_energy)")
        println("  -> Time: $(round(tetris_result.total_runtime, digits=3))s")

        # C. Create Benchmark Result
        bench_res = BenchmarkResult(
            instance_id=instance["instance_id"],
            n_vars=instance["variables"],
            bruteforce_result=bf_result,
            tetris_result=tetris_result
        )
        push!(benchmark_results, bench_res)
    end

    # 5. Save Results
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd-HH-MM-SS")
    results_dir = joinpath(@__DIR__, "..", "results")
    if !isdir(results_dir)
        mkdir(results_dir)
    end

    output_file = joinpath(results_dir, "benchmark_greedy_floor_$(timestamp).json")
    println("\nSaving benchmark results to $output_file")

    open(output_file, "w") do f
        JSON.print(f, benchmark_results, 2)
    end

    println("Benchmark Complete.")

    # 6. Cleanup
    println("\n--- Cleaning up generated datasets ---")
    for file in experiment_files
        rm(file)
        println("Deleted: $file")
    end
    println("Cleanup Complete.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark_experiment()
end
