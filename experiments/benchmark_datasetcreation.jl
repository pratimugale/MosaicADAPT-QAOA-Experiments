
# Set environment variables for single-threaded execution BEFORE loading packages

import Pkg
Pkg.activate(".")

import JSON
import Dates
import Random
using Printf
using PyCall
using LinearAlgebra

# Include project modules
include(joinpath(@__DIR__, "..", "src", "MIS_TETRIS_ADAPT.jl"))
import .MIS_TETRIS_ADAPT: TetrisConfig, BenchmarkResult, BruteForceResult, TetrisResult
import .MIS_TETRIS_ADAPT: run_greedy_tetris, run_bruteforce, parse_cnf_file
import .MIS_TETRIS_ADAPT: ClauseSatisfactionTracer, convert_formula_to_clauses, solve_max_e3sat_exact

# TODO: remove this later
import ADAPT

# Include generic runner
include(joinpath(@__DIR__, "..", "src", "qaoa", "run_tetris.jl"))

"""
    generate_scaling_dataset(n_vars::Int, type::String, num_instances::Int; seed::Int=2000)

Generates a dataset for a specific N and type using the python script.
type: "balanced" or "triangle"
"""
function generate_scaling_dataset(n_vars::Int, type::String, num_instances::Int; seed::Int=2000)
    python_script = joinpath(@__DIR__, "..", "src", "dataset", "satqubolib_max3sat.py")
    venv_python = joinpath(@__DIR__, "..", "venv", "bin", "python3")


    if type == "balanced"
        base_ratio = 3.6
    elseif type == "triangle"
        base_ratio = 3.9
    end

    # Seed the RNG for reproducibility of the ratio variations
    Random.seed!(seed)

    # We must loop to randomize ratio per instance
    for i in 0:(num_instances-1)
        # Randomize ratio +/- 20%
        # rand() gives [0, 1). We want [-0.2, 0.2].
        variation = (rand() * 0.4) - 0.2
        instance_ratio = base_ratio * (1.0 + variation)

        # Call for SINGLE instance with specific ratio
        cmd = `$venv_python $python_script $n_vars $instance_ratio 1 --seed $seed --type $type`
        run(cmd)
    end
end


"""
    run_comprehensive_benchmark()

Main entry point for the comprehensive benchmark (6 configurations + baselines).
"""
function run_comprehensive_benchmark()
    # 1. Configuration
    qubit_counts = [8, 10, 12]
    instances_per_type = 50
    seed = 42

    # Usage: julia script.jl [SEED] [INSTANCES_PER_TYPE]
    if length(ARGS) >= 1
        seed = parse(Int, ARGS[1])
    end
    if length(ARGS) >= 2
        instances_per_type = parse(Int, ARGS[2])
    end

    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd-HH-MM-SS")
    results_dir = joinpath(@__DIR__, "..", "results")
    if !isdir(results_dir)
        mkdir(results_dir)
    end
    output_file = joinpath(results_dir, "benchmark_comprehensive_$(timestamp).json")

    println("=== Starting Comprehensive Benchmark ===")
    println("N values: $qubit_counts")
    println("Instances per setting: $instances_per_type (Balanced and No-Triangle)")
    println("Seed: $seed")
    println("Using $(Base.Threads.nthreads()) threads")
    BLAS.set_num_threads(1)
    println("BLAS threads: $(BLAS.get_num_threads())")
    println("Output: $output_file")
    flush(stdout)

    # Consolidated results container
    final_results = Dict(
        "timestamp" => timestamp,
        "threads" => Base.Threads.nthreads(),
        "config" => nothing,
        "results" => []
    )

    # 2. Main Loop Over N
    for n_vars in qubit_counts
        println("\n>>> Processing N = $n_vars <<<")

        for type in ["balanced", "triangle"]
            println("\n  --- Generating $type dataset for N=$n_vars ---")
            generate_scaling_dataset(n_vars, type, instances_per_type; seed=seed)

            # Locate files and switch directory
            subdir_name = (type == "balanced") ? "balancedsat" : "notrianglesat"
            dataset_dir = joinpath(@__DIR__, "..", "dataset", "satqubolib", subdir_name)

            all_files = readdir(dataset_dir)
            target_files = filter(f -> endswith(f, ".cnf") && occursin("sat_$(n_vars)_vars", f), all_files)

            println("  Found $(length(target_files)) files to process.")

            # Process Execution
            for (idx, cnf_filename) in enumerate(target_files)
                # Create full path
                cnf_path = joinpath(dataset_dir, cnf_filename)

                # Parse
                instance = parse_cnf_file(cnf_path)
                instance["instance_id"] = idx
                formula = MIS_TETRIS_ADAPT.get_formula_as_struct(instance["formula"])

                # --- 1. Compute Baselines (Gurobi) ---
                clauses = convert_formula_to_clauses(formula)
                max_satisfied, _ = solve_max_e3sat_exact(n_vars, clauses)
                gurobi_energy = length(formula) - max_satisfied

                # --- 2. Run Tetris Configurations ---

                # Common Config
                config = TetrisConfig(
                    initial_gamma=0.01,
                    hamiltonian_type="exact",
                    num_shots=1000,
                    layer_stopper_max=n_vars * 2, # Enforce 2N limit
                    energy_floor=gurobi_energy,
                    #floor_stopper_threshold=max(0.01, abs(0.01*gurobi_energy)), 
                    floor_stopper_threshold=0.01,
                    optimizer_tolerance=1e-6,
                    optimizer_max_iterations=1000,
                    slow_stopper_threshold=1e-6,
                    slow_stopper_patience=5,
                    gradient_threshold=1e-6,
                    score_stopper_threshold=1e-6
                )

                # Capture config descriptions once
                captured_config_base = Dict(
                    "initial_gamma" => config.initial_gamma,
                    "hamiltonian_type" => config.hamiltonian_type,
                    "num_shots" => config.num_shots,
                    "layer_stopper_max" => config.layer_stopper_max,
                    "energy_floor" => config.energy_floor,
                    "floor_stopper_threshold" => config.floor_stopper_threshold,
                    "optimizer_tolerance" => config.optimizer_tolerance,
                    "optimizer_max_iterations" => config.optimizer_max_iterations,
                    "slow_stopper_threshold" => config.slow_stopper_threshold,
                    "slow_stopper_patience" => config.slow_stopper_patience,
                    "gradient_threshold" => config.gradient_threshold,
                    "score_stopper_threshold" => config.score_stopper_threshold,
                    "floor_stopper_logic" => "Absolute: 0.01",
                    "layer_limit" => "2 * N",
                    "configurations" => [
                        "Greedy + Double (Old Pool)",
                        "Greedy + Nondiagonal (New Pool)",
                        "KaMIS + Nondiagonal (New Pool)",
                        "KaMIS(10%) + Nondiagonal (New Pool)"
                    ]
                )
                final_results["config"] = captured_config_base

                # Define the 4 runs
                configs_to_run = [
                    # (Method, Pool, PercentTail, Label)
                    (false, "qaoa_double_pool", 0.0, "Greedy_Double_OldPool"),
                    (false, "qaoa_nondiagonal_double_pool", 0.0, "Greedy_NewPool"),
                    (true, "qaoa_nondiagonal_double_pool", 0.0, "KaMIS_NewPool"),
                    (true, "qaoa_nondiagonal_double_pool", 10.0, "KaMIS_10%Tails_NewPool")
                ]

                for (use_kamis, pool_type, pct_tail, label) in configs_to_run
                    t_res = run_tetris(
                        config, instance;
                        pool_type=pool_type,
                        use_kamis=use_kamis,
                        percent_tail_ends_removed=pct_tail
                    )

                    # Record Result
                    res_entry = Dict(
                        "n_vars" => n_vars,
                        "type" => type,
                        "instance_idx" => idx,
                        "seed" => seed,
                        "config_label" => label,
                        "energy_floor" => gurobi_energy,

                        # Params
                        "method" => use_kamis ? "kamis" : "greedy",
                        "pool" => pool_type,
                        "tail_removed_pct" => pct_tail,

                        # Metrics
                        "time" => t_res.total_runtime,
                        "adapt_time" => t_res.adapt_runtime,
                        "energy" => t_res.final_energy,
                        "energy_diff" => t_res.final_energy - config.energy_floor,
                        "iterations" => t_res.num_iterations,
                        "layers" => t_res.num_adapt_layers,
                        "success" => t_res.success,
                        "satisfaction_percent" => t_res.percent_satisfied_clauses,
                        "stop_reason" => t_res.callback_flagged,
                        "first_layer_gradient_sum" => t_res.first_layer_gradient_sum,

                        # Traces (minimal)
                        "adaptation_energies" => t_res.adaptation_energies,
                        "adaptation_clause_satisfaction_percent_trace" => t_res.clause_satisfaction_percent_trace,
                        # "parameter_trace" => t_res.parameter_trace # Can be large
                    )
                    push!(final_results["results"], res_entry)
                end

                # print progress
                if idx % 5 == 0 || idx == length(target_files)
                    print(".")
                    flush(stdout)
                end
            end
            println("") # newline

            # Cleanup Files
            for f in target_files
                rm(joinpath(dataset_dir, f))
            end

            # Intermediate Save 
            open(output_file, "w") do f
                JSON.print(f, final_results, 2)
            end
            println("  -> Saved results for N=$n_vars, Type=$type")

        end

    end

    println("\n=== Comprehensive Benchmark Complete ===")
    println("Results saved to $output_file")
end

run_comprehensive_benchmark()
