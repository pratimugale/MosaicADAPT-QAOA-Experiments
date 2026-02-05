using Test
using PyCall
using JSON
using MIS_TETRIS_ADAPT
include("../src/exact_solvers/gurobi_solver.jl")
using .GurobiSolver

# Import internal helpers from MIS_TETRIS_ADAPT (since implementation details might not be exported)
# We can use getproperty to access non-exported members if needed, or assume they are available 
# if included via MIS_TETRIS_ADAPT module.
# Check imports: MIS_TETRIS_ADAPT includes 'utils/max3sat.jl'.
# Functions like parse_cnf_file, get_formula_as_struct, get_number_of_satisfied_clauses are defined there.
# They are NOT exported by default in MIS_TETRIS_ADAPT.jl (line 6-7 exports specific types).
# So we access them via proper namespace qualification if possible, or include them directly?
# Since they are inside the module, we can access them as MIS_TETRIS_ADAPT.parse_cnf_file.

# Needed for formula conversion since 'ADAPT' might not be in scope directly without using it.
import ADAPT 

@testset "Solver Comparison Test" begin

    # 1. Configuration
    n_vars = 10
    ratio = 4.24
    num_instances = 1
    # Use a random seed
    seed = rand(1000:9999) 
    type_str = "balanced"
    
    # Paths
    project_root = dirname(@__DIR__)
    python_script = joinpath(project_root, "src", "dataset", "satqubolib_max3sat.py")
    venv_python = joinpath(project_root, "venv", "bin", "python3")
    dataset_dir = joinpath(project_root, "dataset", "satqubolib", "balancedsat")

    println("Generating 3-SAT instance with seed $seed...")
    
    # 2. Generate Dataset
    cmd = `$venv_python $python_script $n_vars $ratio $num_instances --seed $seed --type $type_str`
    run(cmd)
    
    # Locate the file
    # Pattern: sat_{n_vars}_vars_{ratio}_ratio_seed{seed}.cnf
    # Or just find the file ending in "seed{seed}.cnf"
    files = readdir(dataset_dir)
    target_file = nothing
    for f in files
        if occursin("seed$seed.cnf", f)
            target_file = joinpath(dataset_dir, f)
            break
        end
    end
    
    @test target_file !== nothing
    println("Testing on instance: $target_file")
    
    try
        # 3. Solve with Gurobi
        println("\n--- Gurobi Solver ---")
        # a. Parse
        instance_dict = MIS_TETRIS_ADAPT.parse_cnf_file(target_file)
        
        # b. Convert to ADAPT Formula
        # Note: get_formula_as_struct is likely internal.
        formula = MIS_TETRIS_ADAPT.get_formula_as_struct(instance_dict["formula"])
        
        # c. Get Approximate Hamiltonian (PauliSum)
        # We need to access ADAPT.Hamiltonians.Max3SAT.get_approximate_hamiltonian
        hamiltonian = ADAPT.Hamiltonians.Max3SAT.get_approximate_hamiltonian(formula, n_vars)
        
        # d. Convert to QUBO
        Q, offset = pauli_to_qubo(hamiltonian, n_vars)
        
        # e. Solve
        energy_gurobi, solution_gurobi = solve_qubo_gurobi(Q, offset)
        
        # f. Check Satisfaction
        # solution_gurobi is Vector{Int} (0/1). 
        # get_number_of_satisfied_clauses expects Vector{Bool}.
        bitstring_gurobi = Vector{Bool}(Bool.(solution_gurobi)) 
        # Wait, typical mapping: 0 -> 1 (spin), 1 -> -1 (spin).
        # But for boolean SAT: 1 is True, 0 is False?
        # Let's verify our Pauli map: Z = 1 - 2x. 
        # If x=1, Z=-1. If x=0, Z=1.
        # In Max3SAT Hamiltonians, typically Z=1 corresponds to False, Z=-1 corresponds to True (or vice-versa).
        # We need to check 'literal_penalty_operator' or 'convert_binary_literal_to_spin_operator'.
        # convert_binary_literal_to_spin_operator: x_n -> (1+Z_n)/2.
        # If x_n is True (1), we want penalty 0? Or reward?
        # Usually standard is: Z=1 (spin up) -> 0, Z=-1 (spin down) -> 1.
        # Check: x -> (1+Z)/2. If Z=1 -> 1. If Z=-1 -> 0.
        # So Z=1 corresponds to "1" in that logic?
        # Actually, let's look at the satisfied clauses check.
        # MIS_TETRIS_ADAPT.get_number_of_satisfied_clauses uses 'bitstring'.
        # It treats bitstring[i] as Boolean val for var i.
        # If solution_gurobi gives x \in {0,1}, we can assume it maps directly.
        # But we must be sure 1 means True.
        
        satisfied_gurobi = MIS_TETRIS_ADAPT.get_number_of_satisfied_clauses(bitstring_gurobi, formula)
        println("Gurobi Found: Energy=$energy_gurobi, Satisfied=$satisfied_gurobi")


        # 4. Solve with RC2 (Python)
        println("\n--- RC2 Solver ---")
        # Add src/exact_solvers to python path
        exact_solvers_path = joinpath(project_root, "src", "exact_solvers")
        pushfirst!(PyVector(pyimport("sys")."path"), exact_solvers_path)
        
        # Import rc2 module
        rc2_module = pyimport("rc2")
        
        # solve_rc2 returns (models, cost)
        # models is list of lists of literals (e.g. [1, -2, 3])
        models_rc2, cost_rc2 = rc2_module.solve_rc2(target_file)
        
        # RC2 cost is usually number of UNSATISFIED clauses (soft clauses weight=1)
        # Total clauses
        num_clauses = length(formula.clauses)
        satisfied_rc2 = num_clauses - cost_rc2
        
        println("RC2 Found: Cost=$cost_rc2, Satisfied=$satisfied_rc2")
        
        # 5. Compare
        # Convert RC2 models to bitstrings
        # RC2 returns literals. If 1 in model -> x1=True. If -1 in model -> x1=False.
        # Gurobi mapping (Z=1-2x, H uses Z):
        # We need to ensure we use the same convention.
        # Assuming x=1 corresponds to True.
        
        best_rc2_energy = Inf
        best_rc2_bitstring = nothing
        
        println("Evaluating RC2 solutions on Approximate Hamiltonian:")
        for model in models_rc2
            # Initialize bitstring (1-based indexing for Julia)
            bits = zeros(Int, n_vars)
            for lit in model
                var = abs(lit)
                # If lit > 0, x=1. If lit < 0, x=0.
                if lit > 0
                    bits[var] = 1
                else
                    bits[var] = 0
                end
            end
            
            # Calculate Energy: x'Qx + offset
            # Manual sparse multiplication
            # energy = bits' * Q * bits + offset
            # But bits is Vector, Q is SparseMatrix. Julia handles this.
            
            e_val = (bits' * Q * bits)[1] + offset
            println("  Model: $model -> Energy: $e_val")
            
            if e_val < best_rc2_energy
                best_rc2_energy = e_val
                best_rc2_bitstring = bits
            end
        end
        
        println("Best RC2 Energy on Approx Hamiltonian: $best_rc2_energy")
        println("Gurobi Energy on Approx Hamiltonian: $energy_gurobi")
        
        # 5. Compare
        # Satisfaction might differ (Approximation Error), so we print it but don't fail hard
        println("Satisfaction Diff: Gurobi=$satisfied_gurobi vs RC2=$satisfied_rc2")
        
        # KEY ASSERTION: Gurobi (Exact Solver for Approx H) must find energy <= RC2 (Approx Solver for Approx H, or Exact for True H)
        # Since Gurobi minimizes THIS specific H, it must be <= any other solution's energy on THIS H.
        @test energy_gurobi <= best_rc2_energy + 1e-9
        
    finally
        # 6. Cleanup
        if target_file !== nothing && isfile(target_file)
            println("Removing test file: $target_file")
            rm(target_file)
        end
    end

end
