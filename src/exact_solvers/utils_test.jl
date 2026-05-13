# Test the convert_formula_to_clauses function
include("utils.jl")
import ADAPT


# create a ADAPT.Hamiltonians.Max3SAT.Types.Formula
formula = ADAPT.Hamiltonians.Max3SAT.Types.Formula([
    ADAPT.Hamiltonians.Max3SAT.Types.Clause((
        ADAPT.Hamiltonians.Max3SAT.Types.Literal(1, false),
        ADAPT.Hamiltonians.Max3SAT.Types.Literal(2, true),
        ADAPT.Hamiltonians.Max3SAT.Types.Literal(3, false)
    )),
    ADAPT.Hamiltonians.Max3SAT.Types.Clause((
        ADAPT.Hamiltonians.Max3SAT.Types.Literal(4, false),
        ADAPT.Hamiltonians.Max3SAT.Types.Literal(5, false),
        ADAPT.Hamiltonians.Max3SAT.Types.Literal(3, true)
    ))
])

clauses = convert_formula_to_clauses(formula)
@assert length(clauses) == 2
@assert clauses[1] == [1, -2, 3]
@assert clauses[2] == [4, 5, -3]
println("Clauses: ", clauses)