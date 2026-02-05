using Test
using PyCall
using JSON
using Statistics
using Random
using MIS_TETRIS_ADAPT
include("../src/exact_solvers/gurobi_solver.jl")
using .GurobiSolver

# Import internal helpers via MIS_TETRIS_ADAPT
import ADAPT 

@testset "Gurobi vs Random Benchmark (100 Instances)" begin

    # 1. Configuration
    n_vars = 15
    ratio = 4.24
    num_instances = 100
    base_seed = 4200
    type_str = "balanced"
    num_random_samples = 1000 # Enough to get a distribution, but likely won't hit ground state for N=10 (2^10=1024) often
    
    # Paths
    project_root = dirname(@__DIR__)
    python_script = joinpath(project_root, "src", "dataset", "satqubolib_max3sat.py")
    venv_python = joinpath(project_root, "venv", "bin", "python3")
    dataset_dir = joinpath(project_root, "dataset", "satqubolib", "balancedsat")

    println("Generating $num_instances 3-SAT instances with base seed $base_seed...")
    
    # 2. Generate Dataset (Batch)
    cmd = `$venv_python $python_script $n_vars $ratio $num_instances --seed $base_seed --type $type_str`
    run(cmd)
    
    # Locate the files
    # We expect 100 files with seeds base_seed to base_seed+99
    target_files = String[]
    all_files = readdir(dataset_dir)
    
    for i in 0:(num_instances-1)
        s = base_seed + i
        match = findfirst(f -> occursin("seed$s.cnf", f), all_files)
        if match !== nothing
            push!(target_files, joinpath(dataset_dir, all_files[match]))
        else
            @warn "Missing file for seed $s"
        end
    end
    
    @test length(target_files) == num_instances
    println("Found $(length(target_files)) instances.")
    
    # Analysis Storage
    better_than_random_min_count = 0
    better_than_random_mean_count = 0
    total_energy_diff_mean = 0.0
    total_satisfied_clauses_gurobi = 0
    total_satisfied_clauses_random_mean = 0.0
    
    try
        println("\n--- Starting Benchmark ---")
        
        for (idx, cnf_path) in enumerate(target_files)
            # a. Setup
            instance_dict = MIS_TETRIS_ADAPT.parse_cnf_file(cnf_path)
            formula = MIS_TETRIS_ADAPT.get_formula_as_struct(instance_dict["formula"])
            hamiltonian = ADAPT.Hamiltonians.Max3SAT.get_approximate_hamiltonian(formula, n_vars)
            
            # b. Gurobi Solve (Ground Truth for Approx Hamiltonian)
            Q, offset = pauli_to_qubo(hamiltonian, n_vars)
            energy_gurobi, solution_gurobi = solve_qubo_gurobi(Q, offset)
            
            # c. Random Sampling
            random_min_energy = Inf
            random_sum_energy = 0.0
            random_satisfied_sum = 0
            
            for _ in 1:num_random_samples
                # Random bitstring (Bool)
                bits = rand(Bool, n_vars)
                
                # Evaluate Energy: x'Qx + offset
                # Julia sparse matrix * Bool vector works (treats Bool as 0/1)
                e_val = (bits' * Q * bits)[1] + offset
                
                if e_val < random_min_energy
                    random_min_energy = e_val
                end
                random_sum_energy += e_val
                
                # Calculate Satisfaction
                sat = MIS_TETRIS_ADAPT.get_number_of_satisfied_clauses(bits, formula)
                random_satisfied_sum += sat
            end
            
            random_mean_energy = random_sum_energy / num_random_samples
            random_mean_satisfied = random_satisfied_sum / num_random_samples
            
            total_satisfied_clauses_random_mean += random_mean_satisfied
            
            # d. Metrics
            # Gurobi should be <= Random Min (it's exact!)
            # Note: For small N=10, random sampling MIGHT hit the exact ground state (chance is 1000/1024 ~ 1.0 if unique).
            # So we expect energy_gurobi <= random_min_energy.
            
            if energy_gurobi <= random_min_energy + 1e-9
                better_than_random_min_count += 1
            end
            
            if energy_gurobi < random_mean_energy
                better_than_random_mean_count += 1
            end
            
            total_energy_diff_mean += (random_mean_energy - energy_gurobi)
            
            # Use solution_gurobi to check satisfaction
            # Z=1-2x. If solution_gurobi is 0/1 x-basis, and get_number_of_satisfied_clauses uses 
            # Vector{Bool} where true=1, false=0.
            # Assuming standard mapping consistency from earlier test.
            bitstring_gurobi = Vector{Bool}(Bool.(solution_gurobi))
            satisfied = MIS_TETRIS_ADAPT.get_number_of_satisfied_clauses(bitstring_gurobi, formula)
            total_satisfied_clauses_gurobi += satisfied

            if idx % 10 == 0
                print(".")
                flush(stdout)
            end
        end
        println("\n")
        
        # 3. Assertions
        println("Gurobi <= Random Min: $better_than_random_min_count / $num_instances")
        println("Gurobi < Random Mean: $better_than_random_mean_count / $num_instances")
        println("Avg Energy Gap (Random Mean - Gurobi): $(total_energy_diff_mean / num_instances)")
        println("Avg Satisfied Clauses (Gurobi): $(total_satisfied_clauses_gurobi / num_instances)")
        println("Avg Satisfied Clauses (Random Mean): $(total_satisfied_clauses_random_mean / num_instances)")
        
        @test better_than_random_min_count == num_instances
        @test better_than_random_mean_count == num_instances
        
    finally
        # 4. Cleanup
        println("\nCleaning up $num_instances generated files...")
        for f in target_files
            rm(f)
        end
    end

end
