include(joinpath(@__DIR__, "..", "src", "MIS_TETRIS_ADAPT.jl"))
using .MIS_TETRIS_ADAPT

using Test

function run_test()
    println("--- Smoke Test: Brute Force Solver ---")

    # 1. Load an instance (e.g., from the dataset we generated)
    dataset_dir = joinpath(@__DIR__, "..", "dataset", "satqubolib", "balancedsat")
    files = filter(f -> endswith(f, ".cnf") && occursin("10_vars", f), readdir(dataset_dir))

    if isempty(files)
        error("No dataset files found. Run dataset generation first.")
    end

    cnf_path = joinpath(dataset_dir, files[1])
    println("Loading instance: $cnf_path")

    # Use qualified name since it's not exported
    instance = MIS_TETRIS_ADAPT.parse_cnf_file(cnf_path)

    # 2. Run Brute Force
    println("Running Brute Force...")
    result = run_bruteforce(instance)

    println("\nResult:")
    println("  Success: $(result.best_satisfaction_count > 0)")
    println("  Best Satisfaction: $(result.best_satisfaction_count)")
    println("  % Satisfied: $(result.percent_satisfied_clauses)")
    println("  Approx Energy: $(result.approx_hamiltonian_energy)")
    println("  Best Solution: $(result.best_solution)")
    println("  Runtime: $(result.execution_time)s")

    # 3. Assertions
    @test result.best_satisfaction_count > 0
    @test length(result.best_solution) == instance["variables"]
    @test abs(result.approx_hamiltonian_energy) > 0.0

    println("\nSmoke Test Passed!")
end

run_test()
