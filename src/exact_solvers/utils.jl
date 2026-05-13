# Utility functions 

import ADAPT

# Function to convert a ADAPT.Hamiltonians.Max3SAT.Types.Formula to a list of clauses
# In case of negation, the variable is negated in the clause.
function convert_formula_to_clauses(
    formula::ADAPT.Hamiltonians.Max3SAT.Types.Formula)::Vector{Vector{Int64}}
    clauses = Vector{Int64}[]
    for clause in formula.clauses
        clause_literals = Int64[]
        for literal in clause.lits
            # If literal.neg is true, it corresponds to a negative integer
            val = literal.neg ? -literal.var : literal.var
            push!(clause_literals, val)
        end
        push!(clauses, clause_literals)
    end
    return clauses
end
    