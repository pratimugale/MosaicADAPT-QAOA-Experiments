
import ADAPT
import PauliOperators: ScaledPauliVector, Pauli, PauliSum, ScaledPauli
import ADAPT.ADAPT_QAOA: QAOAObservable, QAOAAnsatz
import Statistics: mean
import LinearAlgebra: norm

"""
    run_vanilla_qaoa(config::TetrisConfig, instance::Dict;
                     pool_type::String="qaoa_double_pool")

Runs classic one-at-a-time ADAPT-QAOA on a single instance.
"""
function run_vanilla_qaoa(config::TetrisConfig, instance::Dict;
    pool_type::String="qaoa_double_pool",
    instance_filename::String="")
    t_start_total = time()

    instance_id = instance["instance_id"]
    n_vars = instance["variables"]
    @info("Running Vanilla ADAPT-QAOA (pool=$pool_type) on instance $instance_id")

    formula = MIS_TETRIS_ADAPT.get_formula_as_struct(instance["formula"])
    H = ADAPT.Hamiltonians.Max3SAT.get_exact_hamiltonian(formula, n_vars)
    # Note: Using raw ScaledPauliVector (H) instead of QAOAObservable 
    # because QAOAAnsatz requires generators and observable to have the same type, 
    # and QAOAObservable enforces a diagonal constraint that mixers violate.

    # Pool
    local pool
    if pool_type == "qaoa_double_pool"
        pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_double_pool(n_vars)
    elseif pool_type == "qaoa_nondiagonal_double_pool"
        pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_nondiagonal_double_pool(n_vars)
    elseif pool_type == "qaoa_mixer"
        pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_mixer(n_vars)
    else
        error("Unknown pool_type: $pool_type")
    end

    # Ansatz
    qaoa_ansatz = ADAPT.ADAPT_QAOA.QAOAAnsatz(config.initial_gamma, H)
    ψ0 = ones(ComplexF64, 2^n_vars) / sqrt(2^n_vars)
    ψ0 ./= norm(ψ0)

    # Protocols
    adapt = ADAPT.VANILLA
    vqe = ADAPT.OptimOptimizer(:BFGS; g_tol=config.optimizer_tolerance, iterations=config.optimizer_max_iterations)

    trace = ADAPT.Trace()
    callbacks = ADAPT.AbstractCallback[
        ADAPT.Callbacks.Tracer(:energy, :selected_index, :selected_score, :max_pool_gradient, :callback_flagged),
        ADAPT.Callbacks.ParameterTracer(),
        ADAPT.Callbacks.Printer(:energy),
        ADAPT.Callbacks.ScoreStopper(config.score_stopper_threshold),
        ADAPT.Callbacks.LayerStopper(config.layer_stopper_max),
        ClauseSatisfactionTracer(formula, n_vars),
    ]

    if !isnan(config.energy_floor)
        push!(callbacks, ADAPT.Callbacks.FloorStopper(0.01, config.energy_floor))
    end

    # Execution
    t_start_adapt = time()
    success = ADAPT.run!(qaoa_ansatz, trace, adapt, vqe, pool, H, ψ0, callbacks)
    t_adapt = time() - t_start_adapt

    # Extract Results
    final_energy = isempty(trace[:energy]) ? 0.0 : trace[:energy][end]
    n_steps = length(trace[:energy])

    max_pool_gradients = Float64.(get(trace, :max_pool_gradient, Float64[]))
    clause_satisfaction_percent_trace = Float64.(get(trace, :satisfiedclauses, Float64[])) ./ length(formula)
    callback_flagged = get(trace, :callback_flagged, "")

    # Map indices to strings
    selected_indices = get(trace, :selected_index, [])
    selected_operator_strings = [MIS_TETRIS_ADAPT.pauli_op_to_string(pool[i], n_vars) for i in selected_indices]

    # Sampling
    final_state = ADAPT.evolve_state(qaoa_ansatz, ψ0)
    final_state ./= norm(final_state)
    samples_bitmatrix = ADAPT.sample_from_state(final_state, config.num_shots)
    sampled_bitstrings = [Vector{Bool}(samples_bitmatrix[:, i]) for i in 1:size(samples_bitmatrix, 2)]
    sampled_satisfactions = [MIS_TETRIS_ADAPT.get_number_of_satisfied_clauses(bs, formula) for bs in sampled_bitstrings]
    sampled_expected_satisfaction = mean(sampled_satisfactions)
    best_bs, best_sat = MIS_TETRIS_ADAPT.get_best_bitstring_among_sampled_bitstrings(sampled_bitstrings, formula)

    t_total = time() - t_start_total

    return TetrisResult(
        instance_id=instance_id,
        method="adapt_qaoa",
        instance_filename=instance_filename,
        success=success,
        callback_flagged=callback_flagged,
        total_runtime=t_total,
        adapt_runtime=t_adapt,
        final_energy=final_energy,
        hamiltonian_terms=length(H),
        num_clauses=length(formula),
        num_adapt_layers=length(qaoa_ansatz.generators),
        num_iterations=n_steps,
        selected_indices=get(trace, :selected_index, []),
        selected_scores=get(trace, :selected_score, []),
        gamma_values=qaoa_ansatz.γ_parameters,
        beta_values=qaoa_ansatz.β_parameters,
        sampled_expected_satisfaction=sampled_expected_satisfaction,
        sampled_best_satisfaction=best_sat,
        sampled_best_solution=best_bs,
        max_pool_gradients=max_pool_gradients,
        selected_operator_strings=selected_operator_strings,
        clause_satisfaction_percent_trace=clause_satisfaction_percent_trace,
        percent_satisfied_clauses=sampled_expected_satisfaction / length(formula),
        gurobi_energy=Float64(length(formula)),
        approximation_ratio=sampled_expected_satisfaction / length(formula)
    )
end
