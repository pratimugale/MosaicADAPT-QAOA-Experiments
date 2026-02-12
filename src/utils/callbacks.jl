"""
    ClauseSatisfactionTracer()

At each adaptation, identify the number of satisfied clauses and save it as an integer.
For now, this function is to be used only for debugging and testing purposes. We will 
be using it to compare the number of satisfied clauses between different methods like 
the approximate and exact Hamiltonians, and the greedy and Kamis methods.
Note that this function is expensive as it is O(2^N) where N is the number of variables.

"""

struct ClauseSatisfactionTracer <: ADAPT.AbstractCallback
    satisfied_clauses::Vector{Int}
    formula_length::Int
end

function ClauseSatisfactionTracer(formula::ADAPT.Hamiltonians.Max3SAT.Types.Formula, n_vars::Int)
    satisfied_clauses = zeros(Int, 2^n_vars)

    # iterate through all possible configurations
    for i in 0:2^n_vars-1
        # convert i to binary bitstring (Vector{Bool}) of length n_vars
        bitstring = Vector{Bool}(digits(i, base=2, pad=n_vars) .== 1)
        satisfied_clauses[i+1] = get_number_of_satisfied_clauses(bitstring, formula)
    end

    return ClauseSatisfactionTracer(satisfied_clauses, length(formula))
end

function (tracer::ClauseSatisfactionTracer)(
    ::ADAPT.Data, ansatz::ADAPT.AbstractAnsatz, trace::ADAPT.Trace,
    ::ADAPT.AdaptProtocol, ::ADAPT.GeneratorList,
    ::ADAPT.Observable, ψ0::ADAPT.QuantumState,
)
    # Final statevector
    ψ = ADAPT.evolve_state(ansatz, ψ0)

    # Calculate the probability of each state P(x) = |ψ(x)|^2
    prob_dist = abs2.(ψ)

    # Expected number of satisfied clauses = sum(P(x) * satisfied_clauses(x))
    expected_satisfied_clauses = sum(prob_dist .* tracer.satisfied_clauses)
    @info "Expected number of satisfied clauses: $(expected_satisfied_clauses)"
    @info "Percentage of satisfied clauses: $(expected_satisfied_clauses / tracer.formula_length)"

    push!(get!(trace, :satisfiedclauses, Any[]), expected_satisfied_clauses)
    return false
end