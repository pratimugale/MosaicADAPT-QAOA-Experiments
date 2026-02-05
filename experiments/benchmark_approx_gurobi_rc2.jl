import Pkg
Pkg.activate(".")

import JSON
import Dates
import Random
using PyCall
using Printf

# Include project modules
include(joinpath(@__DIR__, "..", "src", "MIS_TETRIS_ADAPT.jl"))
import .MIS_TETRIS_ADAPT: parse_cnf_file
import ADAPT

# Include Gurobi Solver
include(joinpath(@__DIR__, "..", "src", "exact_solvers", "gurobi_solver.jl"))
using .GurobiSolver

# Setup RC2 via PyCall
# Add src/exact_solvers to python path
pushfirst!(PyVector(pyimport("sys")."path"), joinpath(@__DIR__, "..", "src", "exact_solvers"))
const rc2_lib = pyimport("rc2")

"""
    evaluate_hamiltonian(hamiltonian, state::Vector{Int})

Evaluates a Pauli Hamiltonian for a given bitstring state.
State is a vector of 0/1 integers.
"""
function evaluate_hamiltonian(hamiltonian, state::Vector{Int})
    energy = 0.0
    for term in hamiltonian
        coeff = real(term.coeff)
        ops = term.pauli
        
        # Calculate parity of the term for the given state
        # Z_i maps to (-1)^(state[i])? 
        # Wait, Pauli Z: |0> -> |0>, |1> -> -|1> (eigenvalues 1, -1)
        # So Z operator value is +1 if bit is 0, -1 if bit is 1.
        # Z_i = 1 - 2x_i
        
        term_val = 1.0
        
        # Check Z bits
        for i in 1:length(state)
            if (ops.z >> (i-1)) & 1 == 1
                # If bit is 0 -> +1, if bit is 1 -> -1
                term_val *= (state[i] == 0 ? 1.0 : -1.0)
            end
            # If there were X or Y we'd need full state vector, but Approx H should be diagonal (Z only)
        end
        
        energy += coeff * term_val
    end
    return energy
end

"""
    generate_scaling_dataset(n_vars::Int, type::String, num_instances::Int; seed::Int=2000)

Generates a dataset for a specific N and type using the python script.
type: "balanced" or "triangle"
"""
function generate_scaling_dataset(n_vars::Int, type::String, num_instances::Int; seed::Int=2000)
    python_script = joinpath(@__DIR__, "..", "src", "dataset", "satqubolib_max3sat.py")
    venv_python = joinpath(@__DIR__, "..", "venv", "bin", "python3")
    
    # Standard ratio used in other benchmarks
    base_ratio = 4.24 
    
    # We use a simplified generation call here matching benchmark_scaling.jl default
    cmd = `$venv_python $python_script $n_vars $base_ratio $num_instances --seed $seed --type $type`
    run(cmd)
end


function run_benchmark()
    # Configuration
    qubit_counts = [20]
    instances_per_n = 10 # Total 50 instances
    base_seed = 3000     # Different seed than other benchmarks to explore new instances? Or same?
                         # Let's use 3000 to be safe.
    
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd-HH-MM-SS")
    results_dir = joinpath(@__DIR__, "..", "results")
    if !isdir(results_dir)
        mkdir(results_dir)
    end
    output_file = joinpath(results_dir, "benchmark_approx_gurobi_rc2_$(timestamp).json")
    
    println("=== Starting Approx H (Gurobi) vs RC2 Benchmark ===")
    println("N values: $qubit_counts")
    println("Instances per N: $instances_per_n")
    println("Output: $output_file")
    
    final_results = Dict(
        "timestamp" => timestamp,
        "results" => []
    )
    
    type = "balanced"
    
    for n_vars in qubit_counts
        println("\n>>> Processing N = $n_vars <<<")
        
        current_seed = base_seed + (n_vars * 100)
        generate_scaling_dataset(n_vars, type, instances_per_n; seed=current_seed)
        
        dataset_dir = joinpath(@__DIR__, "..", "dataset", "satqubolib", "balancedsat")
        all_files = readdir(dataset_dir)
        
        # Filter for the files we just made
        target_files = filter(f -> endswith(f, ".cnf") && occursin("sat_$(n_vars)_vars", f), all_files)
        
        experiment_files = String[]
        for i in 0:(instances_per_n-1)
            s = current_seed + i
            match = findfirst(f -> occursin("seed$s.cnf", f), target_files)
            if match !== nothing
                push!(experiment_files, joinpath(dataset_dir, target_files[match]))
            end
        end
        
        println("  Found $(length(experiment_files)) files.")
        
        for (idx, cnf_path) in enumerate(experiment_files)
            instance = parse_cnf_file(cnf_path)
            formula = MIS_TETRIS_ADAPT.get_formula_as_struct(instance["formula"])
            
            # 1. Construct Approximate Hamiltonian
            H_approx = ADAPT.Hamiltonians.Max3SAT.get_approximate_hamiltonian(formula, n_vars)
            
            # 2. Solve H_approx with Gurobi
            # The manual says GurobiSolver exports solve_pauli_gurobi
            # We use that as H_approx is a Vector{ScaledPauli}
            gurobi_energy, gurobi_sol = solve_pauli_gurobi(H_approx, n_vars)
            
            # 3. Solve exact instance with RC2
            # rc2_lib.solve_rc2 returns (models, cost)
            # models is a list of lists e.g. [[1, -2, 3], ...]
            rc2_models, rc2_cost = rc2_lib.solve_rc2(cnf_path)
            
            # Take the first model
            if isempty(rc2_models)
                println("  WARNING: RC2 failed to find a model for $cnf_path")
                continue
            end
            
            # RC2 via PyCall might return a Vector{Vector{Int}} (list of lists) OR a Matrix{Int} OR a flat Vector{Int}
            # Logic: If the first element is a number, the whole container is one model (or a matrix of models).
            # If the first element is a collection (vector), then it is a list of models.
            
            local best_model
            # Check if rc2_models[1] is a number (Int64 or similar)
            if rc2_models[1] isa Number
                # Treat the whole container as the stream of literals
                # (If it's a Matrix of multiple models, iterating it all will efficiently overwrite and leave us with the last model's assignment, which is also optimal)
                best_model = rc2_models
            else
                # It's a list of lists, take the first one
                best_model = rc2_models[1]
            end
            
            # Convert DIMACS model to 0/1 vector
            # 1 -> 1 (True), -1 -> 0 (False) ?? 
            # Wait. Let's be careful.
            # IN SAT: x_i=True usually means variable i is 1. x_i=False means 0.
            # IN HAMILTONIAN MAPPING:
            # Usually we verify:
            # Z_i = 1 - 2x_i (where x_i in {0,1})
            # If x_i=0 -> Z_i=1. If x_i=1 -> Z_i=-1.
            
            # Let's map RC2 output to x vector.
            # [1, -2, 3] -> x1=1, x2=0, x3=1.
            
            x_rc2 = zeros(Int, n_vars)
            for lit in best_model
                var_idx = abs(lit)
                if var_idx <= n_vars
                    x_rc2[var_idx] = (lit > 0 ? 1 : 0)
                end
            end
            
            # 4. Evaluate RC2 solution on H_approx
            rc2_approx_energy = evaluate_hamiltonian(H_approx, x_rc2)
            
            # 5. Calculate Satisfied Clauses (New Metric)
            # Convert 0/1 Int vectors to Bool vectors
            gurobi_bool = [x == 1 for x in gurobi_sol]
            rc2_bool = [x == 1 for x in x_rc2]
            
            # Use internal function from MIS_TETRIS_ADAPT (utils/max3sat.jl)
            # Note: access via module name since it might not be exported
            clauses_sat_gurobi = MIS_TETRIS_ADAPT.get_number_of_satisfied_clauses(gurobi_bool, formula)
            clauses_sat_rc2 = MIS_TETRIS_ADAPT.get_number_of_satisfied_clauses(rc2_bool, formula)
            
            # 6. Record
            res_entry = Dict(
                "n_vars" => n_vars,
                "instance_idx" => idx,
                "seed" => current_seed + (idx-1),
                
                # Energy (Approx H Landscape)
                "energy_approx_min_gurobi" => gurobi_energy,
                "energy_approx_rc2" => rc2_approx_energy,
                "diff_energy" => rc2_approx_energy - gurobi_energy,
                
                # Clauses (True MaxSAT Landscape)
                "clauses_sat_gurobi" => clauses_sat_gurobi,
                "clauses_sat_rc2" => clauses_sat_rc2,
                "diff_clauses_sat" => clauses_sat_rc2 - clauses_sat_gurobi,
                
                "rc2_cost_maxsat" => rc2_cost,
                "total_clauses" => length(formula.clauses)
            )
            push!(final_results["results"], res_entry)
            
            if idx % 5 == 0
                print(".")
                flush(stdout)
            end
        end
        println(" Done.")
        
        # Cleanup
        for f in experiment_files
            rm(f)
        end
    end
    
    # Save Final
    open(output_file, "w") do f
        JSON.print(f, final_results, 2)
    end
    println("\nResults saved to $output_file")
end

run_benchmark()
