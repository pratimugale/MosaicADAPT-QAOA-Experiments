
import Pkg
Pkg.activate(".")

import JSON
import Dates
import Random
using Printf
using PyCall

# Include project modules
include(joinpath(@__DIR__, "..", "src", "MIS_TETRIS_ADAPT.jl"))
import .MIS_TETRIS_ADAPT: TetrisConfig, BenchmarkResult, BruteForceResult, TetrisResult
import .MIS_TETRIS_ADAPT: run_greedy_tetris, run_bruteforce, parse_cnf_file
import ADAPT

# Include generic runner
include(joinpath(@__DIR__, "..", "src", "qaoa", "run_tetris.jl"))

# Include Gurobi Solver
include(joinpath(@__DIR__, "..", "src", "exact_solvers", "gurobi_solver.jl"))
using .GurobiSolver

# Setup RC2 via PyCall
pushfirst!(PyVector(pyimport("sys")."path"), joinpath(@__DIR__, "..", "src", "exact_solvers"))
const rc2_lib = pyimport("rc2")

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
    run_comprehensive_benchmark()

Main entry point for the comprehensive benchmark (6 configurations + baselines).
"""
function run_comprehensive_benchmark()
    # 1. Configuration
    qubit_counts = [6]
    instances_per_type = 5
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
    output_file = joinpath(results_dir, "benchmark_comprehensive_$(timestamp).json")

    println("=== Starting Comprehensive Benchmark ===")
    println("N values: $qubit_counts")
    println("Instances per setting: $instances_per_type (Balanced only)")
    println("Base Seed: $base_seed")
    println("Using $(Base.Threads.nthreads()) threads")
    println("Output: $output_file")
    flush(stdout)

    # Consolidated results container
    final_results = Dict(
        "timestamp" => timestamp,
        "threads" => Base.Threads.nthreads(),
        "config" => nothing,
        "results" => []
    )
    
    captured_config_base = nothing

    # 2. Main Loop Over N
    for n_vars in qubit_counts
        println("\n>>> Processing N = $n_vars <<<")

        type = "balanced"
        println("\n  --- Generating $type dataset for N=$n_vars ---")

        current_seed = base_seed + (n_vars * 1000)
        generate_scaling_dataset(n_vars, type, instances_per_type; seed=current_seed)

        # Locate files
        dataset_dir = joinpath(@__DIR__, "..", "dataset", "satqubolib", "balancedsat")
        all_files = readdir(dataset_dir)
        target_files = filter(f -> endswith(f, ".cnf") && occursin("sat_$(n_vars)_vars", f), all_files)

        # Use specific seed-based filtering
        # experiment_files = String[]
        # for i in 0:(instances_per_type-1)
        #     s = current_seed + i
        #     match = findfirst(f -> occursin("seed$s.cnf", f), target_files)
        #     if match !== nothing
        #         push!(experiment_files, joinpath(dataset_dir, target_files[match]))
        #     end
        # end

        println("  Found $(length(target_files)) files to process.")

        # Process Execution
        for (idx, cnf_filename) in enumerate(target_files)
            # Create full path
            cnf_path = joinpath(dataset_dir, cnf_filename)
            
            # Parse
            instance = parse_cnf_file(cnf_path)
            instance["instance_id"] = idx
            formula = MIS_TETRIS_ADAPT.get_formula_as_struct(instance["formula"])

            # --- 1. Compute Baselines (Gurobi & RC2) ---
            # Get Exact Hamiltonian
            H_exact = ADAPT.Hamiltonians.Max3SAT.get_exact_hamiltonian(formula, n_vars)
            
            # Gurobi Energy (Ground Truth)
            gurobi_energy, _ = solve_pauli_gurobi(H_exact, n_vars)

            # RC2 Satisfaction (Ground Truth)
            rc2_models, rc2_cost = rc2_lib.solve_rc2(cnf_path)
            
            pct_satisfied_rc2 = 0.0
            if !isempty(rc2_models)
                best_model_rc2 = (rc2_models[1] isa Number) ? rc2_models : rc2_models[1]
                x_rc2 = zeros(Int, n_vars)
                for lit in best_model_rc2
                    var_idx = abs(lit)
                    if var_idx <= n_vars
                        x_rc2[var_idx] = (lit > 0 ? 1 : 0)
                    end
                end
                rc2_bool = [x == 1 for x in x_rc2]
                num_sat_rc2 = MIS_TETRIS_ADAPT.get_number_of_satisfied_clauses(rc2_bool, formula)
                if length(instance["formula"]) > 0
                    pct_satisfied_rc2 = num_sat_rc2 / length(instance["formula"])
                end
            end
            
            # --- 2. Run 6 Tetris Configurations ---
            
            # Common Config
            base_config = TetrisConfig(
                adapt_type="variable", # placeholder, overwritten by params
                initial_gamma=0.01,
                hamiltonian_type="exact",
                num_shots=1000,
                layer_stopper_max=n_vars*2, # Enforce 2N limit
                energy_floor=gurobi_energy, 
                floor_stopper_threshold=max(0.01, abs(0.01*gurobi_energy)), 
                optimizer_tolerance=1e-4,
                optimizer_max_iterations=10000,
                slow_stopper_threshold=1e-4,
                slow_stopper_patience=10,
                gradient_threshold=1e-4,
                score_stopper_threshold=1e-4
            )

            # Capture config descriptions once
            if captured_config_base === nothing
                captured_config_base = Dict(
                    "initial_gamma" => base_config.initial_gamma,
                    "optimizer_tolerance" => base_config.optimizer_tolerance,
                    "optimizer_max_iterations" => base_config.optimizer_max_iterations,
                    "slow_stopper_threshold" => base_config.slow_stopper_threshold,
                    "slow_stopper_patience" => base_config.slow_stopper_patience,
                    "gradient_threshold" => base_config.gradient_threshold,
                    "score_stopper_threshold" => base_config.score_stopper_threshold,
                    "num_shots" => base_config.num_shots,
                    "floor_stopper_logic" => "max(0.01, 1% of Gurobi Energy)",
                    "layer_limit" => "2 * N",
                    "configurations" => [
                        "Greedy + Double (Old Pool)", 
                        "Greedy + Nondiagonal (New Pool)",
                        "KaMIS + Nondiagonal (New Pool)",
                        "KaMIS(10%) + Nondiagonal (New Pool)"
                    ]
                )
                final_results["config"] = captured_config_base
            end

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
                    base_config, instance;
                    pool_type=pool_type,
                    use_kamis=use_kamis,
                    percent_tail_ends_removed=pct_tail,
                    gurobi_energy=gurobi_energy
                )

                # Record Result
                res_entry = Dict(
                    "n_vars" => n_vars,
                    "type" => type,
                    "instance_idx" => idx,
                    "seed" => current_seed,
                    "config_label" => label,
                    
                    # Params
                    "method" => use_kamis ? "kamis" : "greedy",
                    "pool" => pool_type,
                    "tail_removed_pct" => pct_tail,

                    # Baselines
                    "energy_floor_gurobi" => gurobi_energy,
                    "satisfaction_rc2_percent" => pct_satisfied_rc2,

                    # Metrics
                    "time" => t_res.total_runtime,
                    "adapt_time" => t_res.adapt_runtime,
                    "energy" => t_res.final_energy,
                    "energy_diff" => t_res.final_energy - gurobi_energy,
                    "iterations" => t_res.num_iterations,
                    "layers" => t_res.num_adapt_layers,
                    "success" => t_res.success,
                    "satisfaction_percent" => t_res.percent_satisfied_clauses,
                    "stop_reason" => t_res.callback_flagged,
                    
                    # Traces (minimal)
                    "adaptation_energies" => t_res.adaptation_energies,
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
        println("  -> Saved results for N=$n_vars")
    end

    println("\n=== Comprehensive Benchmark Complete ===")
    println("Results saved to $output_file")
end

run_comprehensive_benchmark()
