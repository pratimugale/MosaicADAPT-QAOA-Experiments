using Test
using PyCall
using JSON
using MIS_TETRIS_ADAPT
include("../src/exact_solvers/gurobi_solver.jl")
using .GurobiSolver

# Import internal helpers
import ADAPT

@testset "Exact Solver Comparison (Gurobi 3-body vs RC2)" begin

    # 1. Configuration
    n_vars = 10
    ratio = 4.24
    num_instances = 1
    seed = 5555
    type_str = "balanced"
    
    # Paths
    project_root = dirname(@__DIR__)
    python_script = joinpath(project_root, "src", "dataset", "satqubolib_max3sat.py")
    venv_python = joinpath(project_root, "venv", "bin", "python3")
    dataset_dir = joinpath(project_root, "dataset", "satqubolib", "balancedsat")

    println("Generating 3-SAT instance with seed $seed...")
    cmd = `$venv_python $python_script $n_vars $ratio $num_instances --seed $seed --type $type_str`
    run(cmd)
    
    # Locate file
    files = readdir(dataset_dir)
    target_file = nothing
    for f in files
        if occursin("seed$seed.cnf", f)
            target_file = joinpath(dataset_dir, f)
            break
        end
    end
    @test target_file !== nothing
    println("Testing on: $target_file")
    
    try
        # 2. Setup Formula
        instance_dict = MIS_TETRIS_ADAPT.parse_cnf_file(target_file)
        formula = MIS_TETRIS_ADAPT.get_formula_as_struct(instance_dict["formula"])
        
        # 3. Get EXACT Hamiltonian (contains 3-body terms)
        # 3. Get EXACT Hamiltonian (contains 3-body terms)
        H_exact = ADAPT.Hamiltonians.Max3SAT.get_exact_hamiltonian(formula, n_vars)
        println("\nDEBUG: First few Hamiltonian terms:")
        for (i, term) in enumerate(H_exact)
             if i > 5; break; end
             println("  $term")
        end
        
        # 4. Solve with Gurobi (Exact Solver with 3-body support)
        println("\n--- Gurobi Exact Solver ---")
        energy_gurobi, solution_gurobi = solve_pauli_gurobi(H_exact, n_vars)
        
        bitstring_gurobi = Vector{Bool}(Bool.(solution_gurobi))
        satisfied_gurobi = MIS_TETRIS_ADAPT.get_number_of_satisfied_clauses(bitstring_gurobi, formula)
        
        println("Gurobi (Exact H): Energy=$energy_gurobi, Satisfied=$satisfied_gurobi")

        # 5. Solve with RC2 (Ground Truth)
        println("\n--- RC2 Solver ---")
        exact_solvers_path = joinpath(project_root, "src", "exact_solvers")
        pushfirst!(PyVector(pyimport("sys")."path"), exact_solvers_path)
        rc2_module = pyimport("rc2")
        
        models_rc2, cost_rc2 = rc2_module.solve_rc2(target_file)
        
        # Calculate RC2 Energy on EXACT Hamiltonian
        # We need to evaluate H_exact for the RC2 solution
        # To avoid manual calculation (complex with 3-body), let's rely on consistency.
        # But we can manually implement evaluate_pauli(H, bits)
        
        function evaluate_pauli(H::Vector{<:ADAPT.PauliOperators.ScaledPauli}, bits::Vector{Int})
            energy = 0.0
            for term in H
                coeff = real(term.coeff)
                ops = term.pauli
                # Expectation of Pauli string on bitstring
                # Z_i -> (1-2x_i)
                term_val = 1.0
                for i in 1:length(bits)
                    if (ops.z >> (i-1)) & 1 == 1
                        # bit i acts on term
                        # bits[i] is 0 or 1.
                        # Z val is 1-2*bits[i] (1 if 0, -1 if 1)
                        term_val *= (1 - 2*bits[i])
                    end
                end
                energy += coeff * term_val
            end
            return energy
        end
        
        best_rc2_energy = Inf
        for model in models_rc2
            bits = zeros(Int, n_vars)
            for lit in model
                var = abs(lit)
                if lit > 0; bits[var] = 1; else; bits[var] = 0; end
            end
            
            e_val = evaluate_pauli(H_exact, bits)
            if e_val < best_rc2_energy
                best_rc2_energy = e_val
            end
        end
        
        println("RC2 Best Energy (on Exact H): $best_rc2_energy")
        
        # 6. Compare
        # Energies should be IDENTICAL (within float error) for Exact Hamiltonian
        @test isapprox(energy_gurobi, best_rc2_energy, atol=1e-5)
        
        # Satisfaction should also match essentially
        num_clauses = length(formula.clauses)
        sat_rc2 = num_clauses - cost_rc2
        
        println("Satisfaction: Gurobi=$satisfied_gurobi vs RC2=$sat_rc2")
        @test satisfied_gurobi == sat_rc2
        
    finally
        if target_file !== nothing && isfile(target_file)
            rm(target_file)
        end
    end

end
