
import Random
import ADAPT
import PauliOperators
import JSON
using Printf

# Include utilities for parsing CNF
include("../src/utils/max3sat.jl")
# Include Max3SAT Hamiltonian definitions
include("../TetrisADAPT.jl/src/hamiltonians/max3sat.jl")

"""
    dict_to_formula(formula_data)

Converts the dictionary representation of a formula (from parse_cnf_file)
into Main.Max3SAT.Types.Formula.
"""
function dict_to_formula(formula_data)
    clauses = Main.Max3SAT.Types.Clause[]

    # formula_data["formula"] is a Vector of Vectors of Dicts
    for clause_list in formula_data["formula"]
        literals = Main.Max3SAT.Types.Literal[]
        for lit_dict in clause_list
            # lit_dict has keys "var" (Int) and "neg" (Bool)
            # In Max3SAT.Types.Literal, var is Int, neg is Bool.
            push!(literals, Main.Max3SAT.Types.Literal(lit_dict["var"], lit_dict["neg"]))
        end
        # Ensure exactly 3 literals per clause for Max3SAT
        if length(literals) != 3
            error("Clause does not have 3 literals: $literals")
        end
        push!(clauses, Main.Max3SAT.Types.Clause((literals[1], literals[2], literals[3])))
    end

    return Main.Max3SAT.Types.Formula(clauses)
end

function run_determinism_test()
    # 1. Load Instance
    dataset_dir = "dataset/satqubolib/balancedsat"
    # Find the generated file (assuming only one from the specific seed/params or just pick the first)
    files = filter(f -> endswith(f, ".cnf"), readdir(dataset_dir))
    if isempty(files)
        error("No CNF files found in $dataset_dir")
    end

    # Sort by modification time to pick the most recently generated one
    sort!(files, by=f -> stat(joinpath(dataset_dir, f)).mtime, rev=true)
    cnf_path = joinpath(dataset_dir, files[1])
    println("Loading instance: $cnf_path")

    instance_dict = parse_cnf_file(cnf_path)
    n_vars = instance_dict["variables"]
    formula = get_formula_as_struct(instance_dict["formula"])

    println("Variables: $n_vars")
    println("Clauses: $(length(formula))")

    println("\nFormula:")
    for (i, clause) in enumerate(formula.clauses)
        lits_str = join(["$(lit.neg ? "!" : "")$(lit.var)" for lit in clause.lits], " OR ")
        println("  C$i: ($lits_str)")
    end

    # 2. Construct Approximate Hamiltonian
    println("Constructing approximate Hamiltonian...")
    H_spv = ADAPT.Hamiltonians.Max3SAT.get_approximate_hamiltonian(formula, n_vars)
    # Wrap in QAOAObservable
    H = ADAPT.ADAPT_QAOA.QAOAObservable(H_spv)

    println("Hamiltonian constructed. Terms: $(length(H_spv))")

    # 3. Determinism Check
    seeds = [123, 456, 789]
    results = []

    println("\nRunning determinism check with seeds: $seeds")

    for seed in seeds
        println("\n--- Run with seed: $seed ---")
        Random.seed!(seed)

        # Setup ADAPT
        # Pool: QAOA Double Pool
        pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_double_pool(n_vars)

        # Ansatz: TetrisQAOAAnsatz
        # Note: 0.1 is gamma0
        ansatz = ADAPT.ADAPT_QAOA.TetrisQAOAAnsatz(0.01, pool, H)

        # Trace
        trace = ADAPT.Trace()

        # Algorithm: Tetris ADAPT
        # threshold
        threshold = 1e-3
        adapt = ADAPT.TETRIS_ADAPT.TETRISADAPT(threshold)

        # Optimizer
        vqe = ADAPT.OptimOptimizer(:BFGS; g_tol=1e-6)

        # Callbacks (Minimal set for this test)
        callbacks = [
            ADAPT.Callbacks.Tracer(:energy, :selected_index, :selected_score),
            ADAPT.Callbacks.ScoreStopper(threshold),
            ADAPT.Callbacks.ParameterStopper(100),
            ADAPT.Callbacks.Printer(:energy),
            ADAPT.Callbacks.SlowStopper(0.5, 3)
        ]

        # Reference state (Superposition)
        ψ0 = ones(ComplexF64, 2^n_vars) / sqrt(2^n_vars)

        # Run
        success = ADAPT.run!(ansatz, trace, adapt, vqe, pool, H, ψ0, callbacks)

        if !success
            println("Warning: Run did not converge according to optimizer check (or stopped early).")
        end

        # Capture results
        final_energy = trace[:energy][end]
        selected_indices = trace[:selected_index]

        println("  Final Energy: $final_energy")
        println("  Selected Indices: $selected_indices")

        push!(results, (seed=seed, energy=final_energy, indices=selected_indices))
    end

    # 4. Verify Consistency
    println("\n--- Verification ---")
    first_result = results[1]
    is_deterministic = true

    for i in 2:length(results)
        res = results[i]
        if res.energy ≈ first_result.energy && res.indices == first_result.indices
            println("Seed $(res.seed) matches Seed $(first_result.seed)")
            println("  Energy: $(res.energy)")
            println("  Operator Indices: $(res.indices)")
        else
            println("MISMATCH: Seed $(res.seed) differs from Seed $(first_result.seed)")
            println("  Energy: $(res.energy) vs $(first_result.energy)")
            println("  Indices: $(res.indices) vs $(first_result.indices)")
            is_deterministic = false
        end
    end

    if is_deterministic
        println("\nRESULT: DETERMINISTIC")
    else
        println("\nRESULT: NON-DETERMINISTIC")ß
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_determinism_test()
end
