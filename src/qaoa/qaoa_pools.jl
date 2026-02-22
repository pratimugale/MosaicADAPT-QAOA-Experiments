# src/qaoa/qaoa_pools.jl
using ADAPT
using ADAPT.ADAPT_QAOA.QAOApools: qaoa_nondiagonal_double_pool
import PauliOperators: ScaledPauliVector, Pauli, PauliSum, ScaledPauli

"""
    qaoa_clause_tailored_pool(n_vars::Int, formula::Vector)

Generates a deterministic pool of 1-qubit, 2-qubit, and instance-tailored 3-qubit operators.
The 3-qubit operators are guaranteed to have globally stable indices for AI predictability.

Returns:
- `local_pool`: The filtered subset of operators to pass to ADAPT.
- `active_mask`: The boolean mask mapping local indices to global indices.
"""
function qaoa_clause_tailored_pool(n_vars::Int, formula)
    # 1. Base Pool (1Q and 2Q non-diagonal operators)
    base_pool = qaoa_nondiagonal_double_pool(n_vars)
    N_base = length(base_pool)

    # 2. Generate Global 3Q Pool
    # We enforce a strict lexicographical ordering of variables (v1 < v2 < v3) 
    # to guarantee absolute determinism of the global index list.
    global_3q_pool = Pauli[]

    # We only want Non-Diagonal operators. The standard QAOA pool convention usually 
    # uses purely X and Y rotations to explore, and Z for the Hamiltonian.
    # To keep the size manageable but highly expressive, we use {X, Y}^3 permutations
    paulis = ['X', 'Y']

    for v1 in 1:(n_vars-2)
        for v2 in (v1+1):(n_vars-1)
            for v3 in (v2+1):n_vars
                # Generate all 8 combinations of X/Y by toggling the Z phase bit
                for z1 in 0:1, z2 in 0:1, z3 in 0:1
                    x_bits = (1 << (v1 - 1)) | (1 << (v2 - 1)) | (1 << (v3 - 1))
                    z_bits = (z1 << (v1 - 1)) | (z2 << (v2 - 1)) | (z3 << (v3 - 1))

                    push!(global_3q_pool, Pauli(z_bits, x_bits, n_vars))
                end
            end
        end
    end

    # The complete global pool
    full_global_pool = vcat(base_pool, global_3q_pool)

    # 3. Filter by Active Clauses
    # Extract all triplet combinations actually present in the formula
    active_triplets = Set{Tuple{Int,Int,Int}}()
    for clause in formula
        # Extract the absolute variable IDs from lit formats and sort them 
        vars = sort([clause.lits[1].var, clause.lits[2].var, clause.lits[3].var])
        push!(active_triplets, (vars[1], vars[2], vars[3]))
    end

    # 4. Build the Active Mask
    # We always include the base 1Q/2Q operators
    active_mask = collect(1:N_base)

    # Append the indices of the 3Q operators that match an active triplet
    for (i, op) in enumerate(global_3q_pool)
        # In PauliOperators, op.pauli.x and op.pauli.z are integer bitmasks 
        # denoting which variables have an X or Z component. The union is the full footprint.
        footprint_mask = op.pauli.x | op.pauli.z

        vars = Int[]
        # Check bits up to n_vars
        for v in 1:n_vars
            if (footprint_mask & (1 << (v - 1))) != 0
                push!(vars, v)
            end
        end

        # Must be exactly 3 vars
        if length(vars) == 3
            if (vars[1], vars[2], vars[3]) in active_triplets
                # i represents the index strictly within the global_3q_pool segment
                push!(active_mask, N_base + i)
            end
        end
    end

    # 5. Extract Local Pool for ADAPT
    # We explicitly type the local_pool as Vector{Vector{ScaledPauli}} because ADAPT relies on 
    # strict typing for its evaluation loops. The base pool already wraps its terms.
    # We must wrap our global 3Q pool operators similarly before combining/returning.
    local_pool = Vector{ScaledPauli{n_vars}}[]

    for i in active_mask
        if i <= N_base
            # Include the natively typed operator from the standard pool
            push!(local_pool, base_pool[i])
        else
            # Wrap the 3Q Pauli in a single-term list with coefficient 1.0 to match ADAPT's expectations
            op = full_global_pool[i]
            push!(local_pool, [ScaledPauli{n_vars}(1.0, op)])
        end
    end

    return local_pool, active_mask
end
