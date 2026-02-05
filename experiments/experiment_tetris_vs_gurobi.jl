
using Test
using JSON
using Statistics
using Random
using Dates
using Printf
using MIS_TETRIS_ADAPT

# Import Gurobi Solver
include("../src/exact_solvers/gurobi_solver.jl")
using .GurobiSolver
import ADAPT 

function run_experiment()
    println("=== Tetris (Approx) vs Gurobi (Approx H) Experiment ===")
    
    # 1. Configuration
    n_vars = 10
    num_instances = 10
    base_seed = 9999
    type_str = "balanced"
    
    # Tetris Config
    tetris_config = TetrisConfig(
        adapt_type="greedy",
        initial_gamma=0.01,
        hamiltonian_type="approximate",
        num_shots=1000,
        layer_stopper_max=10 # enforcing 5 layers
    )
    
    # Paths
    project_root = dirname(@__DIR__)
    python_script = joinpath(project_root, "src", "dataset", "satqubolib_max3sat.py")
    venv_python = joinpath(project_root, "venv", "bin", "python3")
    dataset_dir = joinpath(project_root, "dataset", "satqubolib", "experiment_tetris_gurobi")
    
    if !isdir(dataset_dir)
        mkpath(dataset_dir)
    end

    # 2. Generate Dataset
    println("Generating $num_instances instances (N=$n_vars)...")
    cmd = `$venv_python $python_script $n_vars 4.24 $num_instances --seed $base_seed --type $type_str`
    run(cmd)
    
    # Locate files
    std_dataset_dir = joinpath(project_root, "dataset", "satqubolib", "balancedsat")
    target_files = String[]
    all_files = readdir(std_dataset_dir)
    
    for i in 0:(num_instances-1)
        s = base_seed + i
        match = findfirst(f -> occursin("seed$s.cnf", f), all_files)
        if match !== nothing
            push!(target_files, joinpath(std_dataset_dir, all_files[match]))
        else
            @warn "Missing file for seed $s"
        end
    end
    
    println("Found $(length(target_files)) instances.")
    
    # 3. Main Loop
    results = []
    
    @printf("\n%-10s | %-15s | %-15s | %-15s | %-15s\n", "Instance", "Gurobi Energy", "Tetris Energy", "Gurobi Sat%", "Tetris Sat%")
    println("-"^80)
    
    try
        for (idx, cnf_path) in enumerate(target_files)
            # Parse
            instance_dict = MIS_TETRIS_ADAPT.parse_cnf_file(cnf_path)
            formula = MIS_TETRIS_ADAPT.get_formula_as_struct(instance_dict["formula"])
            num_clauses = length(formula.clauses)
            
            # --- A. Gurobi (Approximate Hamiltonian) ---
            H_approx = ADAPT.Hamiltonians.Max3SAT.get_approximate_hamiltonian(formula, n_vars)
            
            # Use pauli_to_qubo and solve
            Q, offset = pauli_to_qubo(H_approx, n_vars)
            energy_gurobi_raw, sol_gurobi = solve_qubo_gurobi(Q, offset)
            
            # Calculate Satisfaction for Gurobi
            bits_gurobi = Vector{Bool}(Bool.(sol_gurobi))
            sat_gurobi_count = MIS_TETRIS_ADAPT.get_number_of_satisfied_clauses(bits_gurobi, formula)
            sat_gurobi_pct = sat_gurobi_count / num_clauses * 100
            
            
            # --- B. Tetris (Approximate Hamiltonian) ---
            res_tetris = run_greedy_tetris(tetris_config, instance_dict)
            energy_tetris = res_tetris.final_energy
            
            # Tetris satisfaction
            # benchmark_scaling usage:
            sat_tetris_pct = (res_tetris.sampled_expected_satisfaction / num_clauses) * 100
            
            # Print row
            @printf("%-10d | %-15.4f | %-15.4f | %-15.2f | %-15.2f\n", idx, energy_gurobi_raw, energy_tetris, sat_gurobi_pct, sat_tetris_pct)
            flush(stdout)
            
            push!(results, (
                idx=idx, 
                g_en=energy_gurobi_raw, 
                t_en=energy_tetris, 
                g_sat=sat_gurobi_pct, 
                t_sat=sat_tetris_pct
            ))
        end
        
        # 4. Summary
        avg_g_en = mean([r.g_en for r in results])
        avg_t_en = mean([r.t_en for r in results])
        avg_g_sat = mean([r.g_sat for r in results])
        avg_t_sat = mean([r.t_sat for r in results])
        
        println("-"^80)
        @printf("AVERAGE    | %-15.4f | %-15.4f | %-15.2f | %-15.2f\n", avg_g_en, avg_t_en, avg_g_sat, avg_t_sat)
        println("\nComparison Complete.")
        
    finally
        # Cleanup
        for f in target_files
            rm(f)
        end
    end
end

run_experiment()
