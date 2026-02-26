using Test
using MIS_TETRIS_ADAPT
import ADAPT

@testset "ApproxRatioStopper Early Stopping Test" begin
    # 1. Create a dummy formula
    # x1 OR x2 OR x3
    formula_dict = Dict(
        "n_vars" => 3,
        "n_clauses" => 1,
        "clauses" => [[1, 2, 3]]
    )
    formula = MIS_TETRIS_ADAPT.get_formula_as_struct(formula_dict)
    n_vars = 3

    # 2. Setup ADAPT Config
    config = MIS_TETRIS_ADAPT.TetrisConfig(
        initial_gamma=0.01,
        hamiltonian_type="approximate",
        layer_stopper_max=20, # Give it room to run if stopper fails
        gurobi_percent_satisfied_threshold=1.0, # Target 100% satisfaction
        approx_ratio_stopper_threshold=0.5 # A very low threshold to trigger it quickly
    )

    # 3. Create basic ADAPT components
    observable = ADAPT.Hamiltonians.Max3SAT.get_approximate_hamiltonian(formula, n_vars)
    pool = ADAPT.Pools.TetrisMABCA.get_pool(n_vars)

    # Simple Hartree-Fock initial state for 3 qubits
    ψ0 = ADAPT.Statevectors.HartreeFock.get_state(n_vars)

    ansatz = ADAPT.Ansätze.VQE.Minibatch.Ansatz(
        ADAPT.Ansätze.VQE.Minibatch.AnsatzParams[],
        Float64[], # γ_layers
        Float64[]  # β_layers
    )

    adapt = ADAPT.AdaptProtocol()
    trace = ADAPT.Trace()
    callbacks = ADAPT.AbstractCallback[]

    # Add ClauseSatisfactionTracer so ApproxRatioStopper works
    push!(callbacks, ClauseSatisfactionTracer(formula, n_vars))
    # Add our stopper
    # Threshold is 0.5, target is 1.0. This means it will stop as soon as Tetris expects 50% satisfaction
    push!(callbacks, ApproxRatioStopper(config.approx_ratio_stopper_threshold, config.gurobi_percent_satisfied_threshold, length(formula)))

    callbacks_wrapper = ADAPT.Callbacks.List(callbacks)

    # 4. Modify ADAPT parameters to mock iterations or run actual ADAPT
    # We'll just run actual ADAPT since this is a tiny 3-qubit instance.
    ADAPT.run!(
        observable,
        ansatz,
        trace,
        adapt,
        pool,
        ψ0,
        callbacks_wrapper
    )

    n_layers = length(ansatz.γ_layers)

    # With a threshold of 0.5 and just 1 clause, random guessing gives 87.5% satisfaction
    # It should hit this very quickly, likely on layer 1 or 2, well before the old minimum of 10.
    println("ADAPT stopped after $n_layers layers.")

    @test n_layers < 10
    @test n_layers > 0
end
