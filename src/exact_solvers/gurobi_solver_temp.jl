module GurobiSolver

using PyCall
using SparseArrays
import PauliOperators: ScaledPauli, Pauli
import ADAPT  # For Hamiltonian generation access if needed, or we accept the vector directly

export solve_instance_gurobi, solve_qubo_gurobi, pauli_to_qubo

"""
    pauli_to_qubo(hamiltonian::Vector{<:ScaledPauli}, n_vars::Int)

Converts a Pauli Hamiltonian (Ising model) into a QUBO formulation.
Mapping: Z_i -> 1 - 2x_i
Returns: (Q::SparseMatrixCSC, offset::Float64)
where the objective is minimize x'Qx + offset
"""
function pauli_to_qubo(hamiltonian::Vector{<:ScaledPauli}, n_vars::Int)
    # Q matrix (upper triangular or symmetric, we'll just fill symmetric and let solver handle)
    # Gurobi handles x'Qx + c'x + const
    # We will absorb linear terms into the diagonal of Q since x^2 = x for binary variables.
    
    Q = spzeros(Float64, n_vars, n_vars)
    offset = 0.0

    for term in hamiltonian
        coeff = real(term.coeff)
        ops = term.pauli
        
        # Check active Z indices
        # Pauli object has .z (bitmask) or similar. 
        # But Vector{ScaledPauli} usually from PauliOperators.jl
        # Let's inspect active indices.
        
        indices = Int[]
        # We need to find which qubits have Z operators.
        # PauliOperators.Pauli stores x and z bitmasks.
        # But we can iterate 1:n_vars
        if ops.x != 0
            error("Hamiltonian contains non-Z operators (X or Y). Cannot convert to classical QUBO.")
        end

        for i in 1:n_vars
            # Check if Z bit is set for qubit i (bit i-1)
            if (ops.z >> (i-1)) & 1 == 1
                push!(indices, i)
            end
        end
        
        degree = length(indices)
        
        if degree == 0
            # Identity term (Constant)
            offset += coeff
            
        elseif degree == 1
            # Linear term: c * Z_i = c * (1 - 2x_i) = c - 2c x_i
            i = indices[1]
            offset += coeff
            Q[i, i] -= 2.0 * coeff
            
        elseif degree == 2
            # Quadratic term: c * Z_i Z_j = c * (1 - 2x_i)(1 - 2x_j)
            # = c * (1 - 2x_i - 2x_j + 4x_i x_j)
            # = c - 2c x_i - 2c x_j + 4c x_i x_j
            i, j = indices[1], indices[2]
            
            offset += coeff
            Q[i, i] -= 2.0 * coeff
            Q[j, j] -= 2.0 * coeff
            
            # Add cross term. Put 2c on (i,j) and 2c on (j,i) to preserve symmetry?
            # Or just 4c on one? Gurobi typically wants x'Qx.
            # 4c x_i x_j is the term.
            # If Q is symmetric, x'Qx = x_i Q_ij x_j + x_j Q_ji x_i = 2 Q_ij x_i x_j.
            # So we typically set Q_ij = Q_ji = 2c.
            Q[i, j] += 2.0 * coeff
            Q[j, i] += 2.0 * coeff
            
        else
            error("Hamiltonian contains terms with degree > 2 ($degree). Cannot convert to quadratic QUBO without ancillas.")
        end
    end
    
    return Q, offset
end

"""
    solve_qubo_gurobi(Q::AbstractMatrix, offset::Float64; timeout=100.0)

Solves the QUBO problem minimize x'Qx + offset using Gurobi via Python.
"""
function solve_qubo_gurobi(Q::AbstractMatrix, offset::Float64; timeout=100.0)
    gp = pyimport("gurobipy")
    
    n = size(Q, 1)
    
    # Create environment and model (suppress output)
    env = gp.Env(params=Dict("OutputFlag" => 0))
    model = gp.Model("qubo", env=env)
    
    model.setParam("TimeLimit", timeout)
    
    # Add variables
    x = model.addVars(n, vtype=gp.GRB.BINARY, name="x")
    
    # Build Objective
    obj = gp.QuadExpr()
    obj.addConstant(offset)
    
    # We can iterate the sparse matrix Q
    rows, cols, vals = findnz(sparse(Q)) # Ensure sparse
    
    for k in 1:length(vals)
        i = rows[k]
        j = cols[k]
        v = vals[k]
        # In Python gurobipy, indices are 0-based for lists, but x is a tupledict here?
        # x is a tupledict if created with addVars(n). Keys are 0..n-1.
        # Julia i,j are 1-based.
        # Use direct arithmetic update via PyCall/Gurobi overloading
        obj += v * x[i-1] * x[j-1]
    end
    
    model.setObjective(obj, gp.GRB.MINIMIZE)
    
    model.optimize()
    
    # Extract results
    status = model.Status
    if status == gp.GRB.OPTIMAL || status == gp.GRB.TIME_LIMIT
        energy = model.objVal
        
        # Get solution vector
        solution = zeros(Int, n)
        for i in 1:n
            val = x[i-1].X
            # Round to nearest integer (0 or 1)
            solution[i] = round(Int, val)
        end
        
        return energy, solution
    else
        error("Gurobi failed to find a valid solution. Status code: $status")
    end
end

end
