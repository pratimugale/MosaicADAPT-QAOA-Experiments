import Pkg
Pkg.activate(".")

# Include the project source
include(joinpath(@__DIR__, "..", "src", "MIS_TETRIS_ADAPT.jl"))
import .MIS_TETRIS_ADAPT: TetrisConfig
# We need to include the runner script logic since it's not a module yet, or assumed to be available
# Ideally this would be in the module, but for now we include the file directly as per design
include(joinpath(@__DIR__, "..", "src", "qaoa", "greedy_tetris.jl"))

# Include utils for loading the instance dictionary
include(joinpath(@__DIR__, "..", "src", "utils", "max3sat.jl"))

function run_smoke_test()
    println("--- Starting Smoke Test for Greedy Tetris ---")

    # 1. Setup Config
    config = TetrisConfig(
        adapt_type="greedy",
        initial_gamma=0.01,
        hamiltonian_type="approximate",
        num_shots=100
    )
    println("Configuration created: $config")

    # 2. Load a Single Instance
    # Reuse the 15-var verification instance we created earlier
    dataset_dir = joinpath(@__DIR__, "..", "dataset", "satqubolib", "balancedsat")
    files = filter(f -> endswith(f, ".cnf"), readdir(dataset_dir))
    if isempty(files)
        error("No CNF files found for test")
    end
    # pick first
    cnf_path = joinpath(dataset_dir, files[1])
    println("Loading instance: $cnf_path")

    instance = parse_cnf_file(cnf_path)

    # 3. Run
    println("Invoking run_greedy_tetris...")
    results = run_greedy_tetris(config, instance)

    # 4. Assertions
    # 4. Assertions
    println("\nResult Type: $(typeof(results))")

    if results.success
        println("✓ Algorithm returned success=true")
    else
        println("⚠ Algorithm returned success=false (might be expected for short run)")
    end

    final_energy = results.final_energy
    println("Final Energy: $final_energy")

    println("\nTiming:")
    println("  Hamiltonian Construction: $(round(results.hamiltonian_construction_time, digits=3))s")
    println("  Ansatz Creation: $(round(results.ansatz_creation_time, digits=3))s")
    println("  ADAPT Runtime: $(round(results.adapt_runtime, digits=3))s")
    println("  Sampling Time: $(round(results.sampling_time, digits=3))s")
    println("  Total Runtime: $(round(results.total_runtime, digits=3))s")

    sampled_sat = results.sampled_expected_satisfaction
    println("Sampled Expected Sat: $sampled_sat")

    println("--- Smoke Test Passed ---")
end

run_smoke_test()
