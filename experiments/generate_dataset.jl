
# Set environment variables for single-threaded execution BEFORE loading packages

import Pkg
Pkg.activate(".")

import JSON
import Dates
import Random
using Printf
using Base.Threads
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
import .MIS_TETRIS_ADAPT: ClauseSatisfactionTracer, ApproxRatioStopper, convert_formula_to_clauses, solve_max_e3sat_exact
import .MIS_TETRIS_ADAPT: get_formula_as_struct

# TODO: remove this later
import ADAPT

# Include generic runner
include(joinpath(@__DIR__, "..", "src", "qaoa", "run_tetris.jl"))

"""
    pauli_op_to_string(op, n_vars) -> String

Convert a pool operator (Vector{ScaledPauli}) to a human-readable Pauli string.
Uses the x/z bitmask encoding from PauliOperators:
  - x=1, z=0 → X
  - x=1, z=1 → Y  (Y = iXZ)
  - x=0, z=1 → Z

For multi-term operators, joins all terms with "+".
"""
function pauli_op_to_string(op, n_vars::Int)::String
    term_strings = String[]
    for sp in op
        pauli = sp.pauli
        qubit_labels = String[]
        for q in 1:n_vars
            bit = 1 << (q - 1)
            has_x = (pauli.x & bit) != 0
            has_z = (pauli.z & bit) != 0
            if has_x && has_z
                push!(qubit_labels, "Y$q")
            elseif has_x
                push!(qubit_labels, "X$q")
            elseif has_z
                push!(qubit_labels, "Z$q")
            end
        end
        push!(term_strings, isempty(qubit_labels) ? "I" : join(qubit_labels))
    end
    return join(term_strings, "+")
end

"""
    save_pool_operator_map(n_vars, pool_type, output_dir)

Build the operator pool for the given N and pool type, then save a JSON file
mapping each operator index (1-based) to its Pauli string representation.
"""
function save_pool_operator_map(n_vars::Int, pool_type::String, output_dir::String)
    local pool
    if pool_type == "qaoa_double_pool"
        pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_double_pool(n_vars)
    elseif pool_type == "qaoa_nondiagonal_double_pool"
        pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_nondiagonal_double_pool(n_vars)
    else
        println("Warning: unknown pool_type $pool_type, skipping operator map.")
        return
    end

    op_map = Dict{String,String}()
    for (i, op) in enumerate(pool)
        op_map[string(i)] = pauli_op_to_string(op, n_vars)
    end

    map_file = joinpath(output_dir, "pool_operator_map_N$(n_vars)_$(pool_type).json")
    open(map_file, "w") do f
        JSON.print(f, op_map, 2)
    end
    println("Saved pool operator map to $map_file ($(length(op_map)) operators)")
end

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

    initial_gammas = [0.001, 0.01, 0.1]

    # Save pool operator index → Pauli string mapping once (worker 1 only to avoid races)
    if worker_id == 1
        unique_pools = Set([pool_type for (_, pool_type, _, _) in strategies])
        for pt in unique_pools
            save_pool_operator_map(n_vars, pt, results_dir)
        end
    end

    # Locate files
    target_dirs = []
    if dataset_name == "all"
        push!(target_dirs, "balancedsat")
        push!(target_dirs, "notrianglesat")
        push!(target_dirs, "randomsat")
    elseif dataset_name == "both"
        push!(target_dirs, "balancedsat")
        push!(target_dirs, "notrianglesat")
    else
        push!(target_dirs, dataset_name)
    end

    all_target_files = [] # Array of (subdir, filename)

    for subdir in target_dirs
        dataset_dir = joinpath(@__DIR__, "..", "dataset", "satqubolib", subdir)
        if !isdir(dataset_dir)
            println("Warning: Dataset directory not found: $dataset_dir")
            continue
        end

        files = readdir(dataset_dir)
        # Filter for .cnf files and match the N vars
        matched = filter(f -> endswith(f, ".cnf") && occursin("sat_$(n_vars)_vars", f), files)

        for f in matched
            push!(all_target_files, (subdir, f))
        end
    end

    if isempty(all_target_files)
        println("Error: No target files found for N=$n_vars in $(dataset_name)")
        return
    end

    # sort to ensure deterministic order across workers
    sort!(all_target_files, by=x -> x[2])

    # Partition files
    total_files = length(all_target_files)

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
        my_files = all_target_files[start_idx:min(end_idx, total_files)]
    end

    println("Worker $worker_id processing $(length(my_files)) files (indices $start_idx to $end_idx of $total_files)")

    # Process Execution
    # We use @multibreak for loop control flow, but partitioning is handled above.
    @multibreak for (local_idx, (subdir, cnf_filename)) in enumerate(my_files)
        # Create full path
        cnf_path = joinpath(@__DIR__, "..", "dataset", "satqubolib", subdir, cnf_filename)

        # Determine logical type for recording
        type = occursin("triangle", subdir) ? "triangle" : (occursin("random", subdir) ? "random" : "balanced")

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
        gurobi_percent_satisfied = max_satisfied / length(formula)

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
                    layer_stopper_max=3,
                    energy_floor=gurobi_energy,
                    floor_stopper_threshold=0.1,
                    optimizer_tolerance=1e-6,
                    optimizer_max_iterations=1000,
                    slow_stopper_threshold=1e-6,
                    slow_stopper_patience=5,
                    gradient_threshold=1e-6,
                    score_stopper_threshold=1e-6,
                    gurobi_percent_satisfied_threshold=gurobi_percent_satisfied
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
                    "adaptation_clause_satisfaction_percent_trace" => t_res.clause_satisfaction_percent_trace,
                    "sampled_best_satisfaction" => t_res.sampled_best_satisfaction,

                    # Circuit
                    "selected_indices" => t_res.selected_indices,
                    "selected_scores" => t_res.selected_scores,
                    "gamma_values" => t_res.gamma_values,
                    "beta_values" => t_res.beta_values
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
