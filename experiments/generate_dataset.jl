
# Set environment variables for single-threaded execution BEFORE loading packages

import Pkg
Pkg.activate(".")

import JSON
import Dates
import Random
using Printf
using PyCall
using Base.Threads
using LinearAlgebra
using ArgParse
using Multibreak

# Set Julia multithreading based on environment if not already set
# This is typically done before starting Julia, but can be set here if needed.
# For example, if you want to ensure a specific number of threads regardless of environment.
# ENV["JULIA_NUM_THREADS"] = "4" # Uncomment and set if needed

# Include project modules
include(joinpath(@__DIR__, "..", "src", "MIS_TETRIS_ADAPT.jl"))
import .MIS_TETRIS_ADAPT: TetrisConfig, BenchmarkResult, BruteForceResult, TetrisResult
import .MIS_TETRIS_ADAPT: run_greedy_tetris, run_bruteforce, parse_cnf_file
import .MIS_TETRIS_ADAPT: ClauseSatisfactionTracer, convert_formula_to_clauses, solve_max_e3sat_exact
import .MIS_TETRIS_ADAPT: get_formula_as_struct

# TODO: remove this later
import ADAPT

# Include generic runner
include(joinpath(@__DIR__, "..", "src", "qaoa", "run_tetris.jl"))

"""
    parse_commandline()

Parses command line arguments for the script.
"""
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--n_vars"
        help = "Number of variables (N)"
        arg_type = Int
        required = true
        "--num_instances"
        help = "Number of instances total (for checking)"
        arg_type = Int
        default = 50
        "--worker_id"
        help = "ID of this worker (1-based)"
        arg_type = Int
        default = 1
        "--n_workers"
        help = "Total number of workers"
        arg_type = Int
        default = 1
        "--seed"
        help = "Random seed"
        arg_type = Int
        default = 42
        "--output_dir"
        help = "Directory to save results"
        arg_type = String
        default = "results"
        "--dataset_name"
        help = "The subfolder inside dataset/satqubolib to load instances from"
        arg_type = String
        default = "balancedsat"
    end

    return parse_args(s)
end

"""
    main()

Main entry point. 
Performs grid search over initial gamma values and strategies.
Saves two files:
1. All results (every config run)
2. Best results (best config per instance)
Uses @multibreak for dynamic parallelization.
"""
function main()
    args = parse_commandline()

    n_vars = args["n_vars"]
    num_instances = args["num_instances"]
    worker_id = args["worker_id"]
    n_workers = args["n_workers"]
    seed = args["seed"]
    dataset_name = args["dataset_name"]

    BLAS.set_num_threads(1)
    println("BLAS threads: $(BLAS.get_num_threads())")

    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd-HH-MM-SS")

    # Use output_dir from args, fallback to default relative path if "results" is passed/default
    if args["output_dir"] == "results"
        results_dir = joinpath(@__DIR__, "..", "results")
    else
        results_dir = args["output_dir"]
    end

    if !isdir(results_dir)
        mkpath(results_dir)
    end

    println("Environment configurations:")
    println("-> Julia threads: ", Threads.nthreads())
    println("-> BLAS threads:  ", LinearAlgebra.BLAS.get_num_threads())

    # Unique output filenames for this worker
    # Note: Even though partitioning is dynamic via @multibreak, 
    # we still want each worker to write to its own file to avoid locking issues/race conditions.
    output_file_all = joinpath(results_dir, "benchmark_N$(n_vars)_worker$(worker_id)_all_$(timestamp).json")
    output_file_best = joinpath(results_dir, "benchmark_N$(n_vars)_worker$(worker_id)_best_$(timestamp).json")

    println("=== Starting Benchmark Worker $worker_id / $n_workers ===")
    println("N: $n_vars")
    println("Seed: $seed")
    println("Output All: $output_file_all")
    println("Output Best: $output_file_best")
    flush(stdout)

    # Consolidated results containers
    final_results_all = Dict(
        "timestamp" => timestamp,
        "n_vars" => n_vars,
        "worker_id" => worker_id,
        "results" => []
    )

    final_results_best = Dict(
        "timestamp" => timestamp,
        "n_vars" => n_vars,
        "worker_id" => worker_id,
        "results" => []
    )

    # Define Grid
    # Strategies: (use_kamis, pool_type, pct_tail, label_base)
    strategies = [
        (false, "qaoa_nondiagonal_double_pool", 0.0, "Greedy_NewPool"),
        (true, "qaoa_nondiagonal_double_pool", 0.0, "KaMIS_NewPool")
    ]

    initial_gammas = [0.001, 0.01, 0.1, 0.5, 1.0]

    # Locate files
    # Only "balanced" for now as per plan
    type = dataset_name
    dataset_dir = joinpath(@__DIR__, "..", "dataset", "satqubolib", dataset_name)

    if !isdir(dataset_dir)
        println("Error: Dataset directory not found: $dataset_dir")
        return
    end

    all_files = readdir(dataset_dir)
    # Filter for .cnf files and match the N vars
    target_files = filter(f -> endswith(f, ".cnf") && occursin("sat_$(n_vars)_vars", f), all_files)

    # sort to ensure deterministic order across workers
    sort!(target_files)

    # Partition files
    total_files = length(target_files)

    # Simple chunking
    files_per_worker = div(total_files, n_workers)
    extras = total_files % n_workers

    # Calculate start and end indices
    # Distribute extras among first 'extras' workers
    if worker_id <= extras
        start_idx = (worker_id - 1) * (files_per_worker + 1) + 1
        end_idx = start_idx + files_per_worker
    else
        start_idx = extras * (files_per_worker + 1) + (worker_id - extras - 1) * files_per_worker + 1
        end_idx = start_idx + files_per_worker - 1
    end

    if start_idx > total_files
        my_files = []
    else
        my_files = target_files[start_idx:min(end_idx, total_files)]
    end

    println("Worker $worker_id processing $(length(my_files)) files (indices $start_idx to $end_idx of $total_files)")

    # Process Execution
    # We use @multibreak for loop control flow, but partitioning is handled above.
    @multibreak for (local_idx, cnf_filename) in enumerate(my_files)
        # Create full path
        cnf_path = joinpath(dataset_dir, cnf_filename)

        # Instance ID - global index matches the original list
        global_idx = start_idx + local_idx - 1

        println("Worker $worker_id Processing $cnf_filename (Global ID: $global_idx)")

        # Parse
        instance = parse_cnf_file(cnf_path)
        instance["instance_id"] = global_idx
        formula = get_formula_as_struct(instance["formula"])

        # --- 1. Compute Baselines (Gurobi) ---
        clauses = convert_formula_to_clauses(formula)
        max_satisfied, _ = solve_max_e3sat_exact(n_vars, clauses)
        gurobi_energy = length(formula) - max_satisfied

        # Store results for this instance
        instance_results = []

        # --- 2. Grid Search ---
        for gamma in initial_gammas
            for (use_kamis, pool_type, pct_tail, label_base) in strategies

                label = "$(label_base)_g$(gamma)"

                config = TetrisConfig(
                    initial_gamma=gamma,
                    hamiltonian_type="exact",
                    num_shots=1000,
                    layer_stopper_max=n_vars * 2,
                    energy_floor=gurobi_energy,
                    floor_stopper_threshold=0.1,
                    optimizer_tolerance=1e-6,
                    optimizer_max_iterations=1000,
                    slow_stopper_threshold=1e-6,
                    slow_stopper_patience=5,
                    gradient_threshold=1e-6,
                    score_stopper_threshold=1e-6
                )

                t_res = run_tetris(
                    config, instance;
                    pool_type=pool_type,
                    use_kamis=use_kamis,
                    percent_tail_ends_removed=pct_tail
                )

                # Create Result Entry
                res_entry = Dict(
                    "n_vars" => n_vars,
                    "type" => type,
                    "instance_idx" => global_idx,
                    "filename" => cnf_filename,
                    "seed" => seed,
                    "config_label" => label,
                    "energy_floor" => gurobi_energy,
                    "max_satisfied_gurobi" => max_satisfied,
                    "num_clauses" => length(clauses),

                    # Params
                    "initial_gamma" => gamma,
                    "method" => use_kamis ? "kamis" : "greedy",
                    "pool" => pool_type,
                    "hamiltonian_type" => config.hamiltonian_type,
                    "num_shots" => config.num_shots,
                    "layer_stopper_max" => config.layer_stopper_max,
                    "floor_stopper_threshold" => config.floor_stopper_threshold,
                    "optimizer_tolerance" => config.optimizer_tolerance,
                    "optimizer_max_iterations" => config.optimizer_max_iterations,
                    "slow_stopper_threshold" => config.slow_stopper_threshold,
                    "slow_stopper_patience" => config.slow_stopper_patience,
                    "gradient_threshold" => config.gradient_threshold,
                    "score_stopper_threshold" => config.score_stopper_threshold,
                    "percent_tail_ends_removed" => pct_tail,
                    "time" => t_res.total_runtime,
                    "adapt_time" => t_res.adapt_runtime,
                    "tetris_final_energy" => t_res.final_energy,
                    "iterations" => t_res.num_iterations,
                    "layers" => t_res.num_adapt_layers,
                    "success" => t_res.success,
                    "tetris_satisfaction_percent" => t_res.percent_satisfied_clauses,
                    "stop_reason" => t_res.callback_flagged,

                    # Traces
                    "adaptation_energies" => t_res.adaptation_energies,
                    "adaptation_clause_satisfaction_percent_trace" => t_res.clause_satisfaction_percent_trace
                )

                push!(instance_results, res_entry)
                push!(final_results_all["results"], res_entry)

            end # strategies
        end # gammas

        # --- 3. Find Best Result ---
        sort!(instance_results, by=x -> (-x["tetris_satisfaction_percent"], x["layers"]))

        best_res = instance_results[1]
        push!(final_results_best["results"], best_res)

        # Print progress
        print(".")
        flush(stdout)

        # Intermediate Save (saves valid JSON array structure each time)
        if local_idx % 5 == 0 || local_idx == length(my_files)
            open(output_file_all, "w") do f
                JSON.print(f, final_results_all, 2)
            end
            open(output_file_best, "w") do f
                JSON.print(f, final_results_best, 2)
            end
        end

    end # @multibreak files

    # Final Save
    open(output_file_all, "w") do f
        JSON.print(f, final_results_all, 2)
    end
    open(output_file_best, "w") do f
        JSON.print(f, final_results_best, 2)
    end

    println("\n=== Worker $worker_id Complete ===")
    println("Saved $(length(final_results_all["results"])) total runs.")
    println("Saved $(length(final_results_best["results"])) best runs.")
end

main()
