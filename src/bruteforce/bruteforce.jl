import ADAPT
import ADAPT.ADAPT_QAOA: QAOAObservable

"""
    run_bruteforce(instance::Dict)

Exhaustively searches the solution space (2^n) to find the exact optimal solution for the Max-3-SAT instance
with respect to the APPROXIMATE Hamiltonian energy.
Calculates the percent satisfied clauses and the energy of this minimum energy solution.

Returns a `BruteForceResult`.
"""
function run_bruteforce(instance::Dict)
    t_start = time()

    instance_id = instance["instance_id"]
    n_vars = instance["variables"]
    println("Running BruteForce on instance $instance_id with $n_vars variables")

    # 1. Parse Formula
    formula = get_formula_as_struct(instance["formula"])

    # 2. Construct Hamiltonian (Approximate)
    # We construct it once to evaluate energy for all bitstrings
    H_spv_vector = ADAPT.Hamiltonians.Max3SAT.get_approximate_hamiltonian(formula, n_vars)
    H_obs = ADAPT.ADAPT_QAOA.QAOAObservable(H_spv_vector)

    # 3. Preparation
    min_energy = Inf
    best_bitstring = Vector{Bool}(undef, n_vars)
    best_sat_count = -1

    ψ_buffer = zeros(ComplexF64, 2^n_vars)

    # 4. Iterate all possible bitstrings to find MINIMUM ENERGY
    for i in 0:(2^n_vars-1)
        # Convert integer to bitstring (LSB first for indexing in our convention)
        candidate_bs = digits(Bool, i, base=2, pad=n_vars)

        # Set buffer for basis state |i> (Little Endian: index i+1)
        fill!(ψ_buffer, 0.0)
        ψ_buffer[i+1] = 1.0

        # Evaluate energy
        # Note: evaluate returns real for Hermitian
        energy = ADAPT.evaluate(H_obs, ψ_buffer)

        # We strictly want the lowest energy solution
        # If multiple states have the same min energy, we just pick the first one encountered
        if energy < min_energy
            min_energy = energy
            best_bitstring = candidate_bs
            # We calculate satisfaction for this specific candidate 
            # effectively "saving" it as the user requested
            best_sat_count = get_number_of_satisfied_clauses(candidate_bs, formula)
        end
    end

    # 5. Calculate Stats for the Best (Min Energy) Solution
    num_clauses = length(instance["formula"])
    pct_satisfied = 0.0
    if num_clauses > 0
        pct_satisfied = best_sat_count / num_clauses
    end

    execution_time = time() - t_start

    return BruteForceResult(
        instance_id=instance_id,
        method="bruteforce",
        best_solution=best_bitstring,
        best_satisfaction_count=best_sat_count,
        percent_satisfied_clauses=pct_satisfied,
        approx_hamiltonian_energy=min_energy,
        execution_time=execution_time
    )
end
