import Pkg
Pkg.activate(".")

import JSON
import Dates
import Random

# Include project modules
include(joinpath(@__DIR__, "..", "src", "MIS_TETRIS_ADAPT.jl"))
import .MIS_TETRIS_ADAPT: TetrisConfig, TetrisResult

# Include the runner logic
include(joinpath(@__DIR__, "..", "src", "qaoa", "greedy_tetris.jl"))

# Include utils for loading instances
include(joinpath(@__DIR__, "..", "src", "utils", "max3sat.jl"))

"""
    generate_dataset_via_cli(num_instances::Int; seed::Int=123)

Generates a fresh dataset of 10-variable balanced Max-3-SAT instances using the Python script.
"""
function generate_dataset_via_cli(num_instances::Int; seed::Int=123)
    println("--- Generating Dataset ---")
    python_script = joinpath(@__DIR__, "..", "src", "dataset", "satqubolib_max3sat.py")
    venv_python = joinpath(@__DIR__, "..", "venv", "bin", "python3")

    # Arguments matches the python script: num_variables ratio num_instances --seed ... --type balanced
    # We use 10 variables, 4.24 ratio (balanced default approx)
    cmd = `$venv_python $python_script 10 4.24 $num_instances --seed $seed --type balanced`

    println("Running command: $cmd")
    run(cmd)
    println("Dataset generation complete.")
end

"""
    run_dataset_experiment()

Main entry point: generates dataset, runs experiments, saves results.
"""
function run_dataset_experiment()
    # 1. Configuration
    # Using a random seed for dataset generation to ensure variety if run multiple times
    # Or fixed for reproducibility. Let's use a fixed base seed.
    dataset_seed = 999
    num_instances = 10

    # Experiment Config
    config = TetrisConfig(
        adapt_type="greedy",
        initial_gamma=0.01,
        hamiltonian_type="approximate",
        num_shots=1000
    )

    # 2. Generate Dataset
    generate_dataset_via_cli(num_instances, seed=dataset_seed)

    # 3. Locate generated files
    # The python script outputs to dataset/satqubolib/balancedsat/
    # We want to pick the files that match our generation (or just all valid ones, assuming clean slate or overwrite)
    # The python script names them: sat_10_vars_4.24_ratio_seed{SEED+i}.cnf
    dataset_dir = joinpath(@__DIR__, "..", "dataset", "satqubolib", "balancedsat")

    # Simple strategy: Process all 10-var files in that directory to be safe, 
    # or strictly filter by the expected filenames. Let's process all 10-var files.
    all_files = readdir(dataset_dir)
    target_files = filter(f -> endswith(f, ".cnf") && occursin("10_vars", f), all_files)

    # Sort to ensure deterministic order (though redundant with struct storage)
    sort!(target_files)

    # We might have more files if previous runs exist. 
    # To be precise, we can filter by the specific seeds we just generated: 999 to 1008
    expected_seeds = [dataset_seed + i for i in 0:(num_instances-1)]
    experiment_files = String[]

    for seed in expected_seeds
        # Find file matching this seed
        # Pattern: ...seed{seed}.cnf
        match = findfirst(f -> occursin("seed$seed.cnf", f), target_files)
        if match !== nothing
            push!(experiment_files, joinpath(dataset_dir, target_files[match]))
        else
            println("Warning: Could not find generated file for seed $seed")
        end
    end

    if isempty(experiment_files)
        error("No matching experiment files found in $dataset_dir")
    end

    println("Found $(length(experiment_files)) instances to process.")

    # 4. Run Loop
    results_array = TetrisResult[]

    for (i, cnf_path) in enumerate(experiment_files)
        println("\nProcessing $i/$(length(experiment_files)): $(basename(cnf_path))")
        instance = parse_cnf_file(cnf_path)

        # Override instance_id with our loop index/seed for clarity if needed, 
        # but let's keep what's in the file or just trust the results struct.

        result = run_greedy_tetris(config, instance)

        # Calculate and store instance stats
        # instance["formula"] is a vector of clauses (which are vectors of dicts)
        num_clauses = length(instance["formula"])
        pct_satisfied = 0.0
        if num_clauses > 0
            pct_satisfied = result.sampled_expected_satisfaction / num_clauses
        end

        result.num_clauses = num_clauses
        result.percent_satisfied_clauses = pct_satisfied

        println("  -> Clauses: $num_clauses, % Satisfied: $(round(pct_satisfied, digits=4))")

        push!(results_array, result)
    end

    # 5. Save Results
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd-HH-MM-SS")
    results_dir = joinpath(@__DIR__, "..", "results")
    if !isdir(results_dir)
        mkdir(results_dir)
    end

    output_file = joinpath(results_dir, "greedy_dataset_$(timestamp).json")

    println("\nSaving results to $output_file")

    # Convert structs to Dicts/JSON-compatible format
    # JSON.lower should handle it if defined, otherwise we explicitly map
    # We added JSON.lower/struct support in src/results.jl so `JSON.print` should work directly
    # provided we pass the array.

    open(output_file, "w") do f
        JSON.print(f, results_array, 2)
    end

    println("Experiment Complete.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_dataset_experiment()
end
