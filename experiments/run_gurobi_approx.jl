using Pkg
Pkg.activate(".")
using PyCall
using PauliOperators

# Bypass ADAPT package loading and include Max3SAT directly
# Max3SAT.jl is at TetrisADAPT.jl/src/hamiltonians/max3sat.jl
include(joinpath(@__DIR__, "..", "TetrisADAPT.jl", "src", "hamiltonians", "max3sat.jl"))
using .Max3SAT

gp = pyimport("gurobipy")

"""
    solve_approx_hamiltonian_gurobi(n_vars, clauses)

Solves the Max3SAT instance using the APPROXIMATE Hamiltonian constructed by TetrisADAPT.
"""
function solve_approx_hamiltonian_gurobi(n_vars, clauses_list)
    # 1. Convert clauses to TetrisADAPT Formula
    adapt_clauses = Vector{Max3SAT.Types.Clause}()
    
    for c in clauses_list
        lits = []
        for l in c
            var_idx = abs(l)
            is_neg = (l < 0)
            push!(lits, Max3SAT.Types.Literal(var_idx, is_neg))
        end
        if length(lits) != 3
            error("Clause must have exactly 3 literals")
        end
        push!(adapt_clauses, Max3SAT.Types.Clause((lits[1], lits[2], lits[3])))
    end
    
    formula = Max3SAT.Types.Formula(adapt_clauses)
    
    # 2. Get Approximate Hamiltonian
    hamiltonian = Max3SAT.get_approximate_hamiltonian(formula, n_vars)
    
    println("Constructed Approximate Hamiltonian with ", length(hamiltonian), " terms.")

    # 3. Setup Gurobi Model
    model = gp.Model("ApproxHamiltonian_GroundState")
    model.setParam("OutputFlag", 0)
    
    x = model.addVars(n_vars, vtype=gp.GRB.BINARY, name="x")
    
    obj_expr = gp.QuadExpr()
    
    for term in hamiltonian
        coeff = real(term.coeff)
        z_mask = term.pauli.z
        
        indices = Int[]
        for i in 1:64
            if ((z_mask >> (i-1)) & 1) == 1
                push!(indices, i)
            end
        end
        
        if abs(coeff) > 1e-10
            # println("Term: coeff=$coeff, indices=$indices")
        end

        # Add term to objective
        if isempty(indices)
            obj_expr += coeff
            
        elseif length(indices) == 1
            i = indices[1]
            var_i = x[i-1]
            # Z_i = 1 - 2x_i
            obj_expr += coeff * (1 - 2*var_i)
            
        elseif length(indices) == 2
            i = indices[1]
            j = indices[2]
            var_i = x[i-1]
            var_j = x[j-1]
            
            # Z_i Z_j = 1 - 2x_i - 2x_j + 4x_i x_j
            term_val = 1 - 2*var_i - 2*var_j + 4*var_i*var_j
            obj_expr += coeff * term_val
            
        else
            error("Approximate Hamiltonian should be 2-local. Found indices: $indices")
        end
    end
    
    model.setObjective(obj_expr, gp.GRB.MINIMIZE)
    model.optimize()
    
    if model.Status == gp.GRB.OPTIMAL
        ground_energy = model.ObjVal
        ground_state = [round(Int, x[i].X) for i in 0:n_vars-1]
        return ground_energy, ground_state, hamiltonian
    else
        error("Gurobi did not find an optimal solution. Status: ", model.Status)
    end
end

"""
    evaluate_energy(bitstring, hamiltonian)

Calculates the energy of a bitstring [x0, x1, ...] where xi ∈ {0, 1}
using the Pauli Hamiltonian terms.
Mapping: Z_i -> (-1)^(x_i). (0 -> 1, 1 -> -1)
"""
function evaluate_energy(bitstring, hamiltonian)
    energy = 0.0
    for term in hamiltonian
        coeff = real(term.coeff)
        z_mask = term.pauli.z
        
        # Calculate parity of the term for the given bitstring
        # Term value is product of Z_i for all i in mask.
        # Z_i = 1 if bit[i] == 0, -1 if bit[i] == 1.
        # Product is (-1)^(sum of bits in mask)
        
        parity = 0
        for i in 1:64
            if ((z_mask >> (i-1)) & 1) == 1
                # Bitstring is 0-indexed in problem but 1-based in Julia
                # bitstring is Vector so bitstring[i] is the i-th variable
                if i > length(bitstring)
                     # If mask has bit outside range, usually error, but maybe unused var?
                     # Treat as 0 (Identity) -> Z=1
                else
                    parity += bitstring[i]
                end
            end
        end
        
        sign = (parity % 2 == 0) ? 1.0 : -1.0
        energy += coeff * sign
    end
    return energy
end

"""
    count_satisfied_clauses(bitstring, clauses)

Counts how many clauses are satisfied by the bitstring.
"""
function count_satisfied_clauses(bitstring, clauses)
    satisfied_count = 0
    for clause in clauses
        # Clause is [l1, l2, l3]
        # Satisfied if AT LEAST ONE literal is true.
        is_sat = false
        for lit in clause
            var_idx = abs(lit) # 1-based index
            is_neg = (lit < 0)
            
            val = bitstring[var_idx] # 0 or 1
            
            # Literal is true if:
            # (val == 1 and not neg) OR (val == 0 and neg)
            if (val == 1 && !is_neg) || (val == 0 && is_neg)
                is_sat = true
                break
            end
        end
        if is_sat
            satisfied_count += 1
        end
    end
    return satisfied_count
end

# --- Test ---
# Case from earlier
clauses = [[-1, -2, -3], [1, -2, 3], [1, 2, -3], [-1, 2, 3], [-1, -2, 3], [-1, 2, -3], [1, 2, 3], [1, -2, -3], [1, 2, 4], [-2, -3, -4]]

n_vars = 4

println("\n--- Gurobi Optimization ---")
energy_opt, state_opt, H_approx = solve_approx_hamiltonian_gurobi(n_vars, clauses)
println("Approx Ground State Energy (Gurobi): ", energy_opt)
println("Optimal Configuration: ", state_opt)

# Verify Gurobi Result with Manual Calculation
calc_energy_opt = evaluate_energy(state_opt, H_approx)
sat_opt = count_satisfied_clauses(state_opt, clauses)
println("Calculated Energy (Manual): ", calc_energy_opt)
println("Satisfied Clauses: ", sat_opt, "/", length(clauses))


println("\n--- Test Comparison with Other Bitstrings ---")
test_states = [
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1],
]

for s in test_states
    e = evaluate_energy(s, H_approx)
    sat = count_satisfied_clauses(s, clauses)
    println("State: $s | Energy: $(round(e, digits=4)) | Satisfied: $sat")
end
