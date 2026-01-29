# Run tetris adapt on Max3SAT dataset
import ADAPT
import PauliOperators: ScaledPauliVector, Pauli, PauliSum, ScaledPauli
import ADAPT.ADAPT_QAOA: QAOAObservable
import Statistics: mean
import LinearAlgebra: norm

"""
    run_greedy_tetris(config::TetrisConfig, instance::Dict)

Runs the Greedy Tetris ADAPT algorithm on a single Max-3-SAT instance.
Returns a dictionary containing the results.
"""
function run_greedy_tetris(config::TetrisConfig, instance::Dict)
    t_start_total = time()

    # Extract instance details
    instance_id = instance["instance_id"]
    n_vars = instance["variables"]
    println("Running greedy tetris on instance $instance_id with $(n_vars) variables")

    # 1. Parse Formula
    formula = get_formula_as_struct(instance["formula"])

    # 2. Results container (now just using variables effectively)

    # 3. Construct Hamiltonian
    t_start_ham = time()
    if config.hamiltonian_type == "exact"
        H_spv_vector = ADAPT.Hamiltonians.Max3SAT.get_exact_hamiltonian(formula, n_vars)
    elseif config.hamiltonian_type == "approximate"
        H_spv_vector = ADAPT.Hamiltonians.Max3SAT.get_approximate_hamiltonian(formula, n_vars)
    else
        error("Unknown hamiltonian_type: $(config.hamiltonian_type)")
    end
    t_ham = time() - t_start_ham

    # Wrap in QAOAObservable
    H = ADAPT.ADAPT_QAOA.QAOAObservable(H_spv_vector)

    # 4. Create Ansatz & Pool
    t_start_ansatz = time()
    pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_double_pool(n_vars)
    qaoa_ansatz = ADAPT.ADAPT_QAOA.TetrisQAOAAnsatz(config.initial_gamma, pool, H)
    t_ansatz = time() - t_start_ansatz

    # 5. Initial State (Superposition)
    ψ0 = ones(ComplexF64, 2^n_vars) / sqrt(2^n_vars) # normalized

    # The normalization step looks redundant, but keeping it for now as it is in 
    #  the other examples like https://github.com/KarunyaShirali/ADAPT.jl/blob/6fa330f6192eabb159acce8fd58a58ef76228232/test/qaoa_tetris.jl#L81 
    ψ0 /= norm(ψ0)

    # 6. Setup ADAPT Algorithm
    adapt = ADAPT.TETRIS_ADAPT.TETRISADAPT(config.gradient_threshold)

    vqe = ADAPT.OptimOptimizer(:BFGS;
        g_tol=config.optimizer_tolerance,
        iterations=config.optimizer_max_iterations
    )

    trace = ADAPT.Trace()

    callbacks = [
        ADAPT.Callbacks.Tracer(:energy, :selected_index, :selected_score, :sum_gradients, :callback_flagged),
        ADAPT.Callbacks.ParameterTracer(),
        ADAPT.Callbacks.Printer(:energy),
        ADAPT.Callbacks.ScoreStopper(config.score_stopper_threshold),
        ADAPT.Callbacks.ParameterStopper(config.parameter_stopper_max),
        ADAPT.Callbacks.LayerStopper(config.layer_stopper_max),
        ADAPT.Callbacks.SlowStopper(config.slow_stopper_threshold, config.slow_stopper_patience),
        ADAPT.Callbacks.FloorStopper(config.floor_stopper_threshold, config.energy_floor)
    ]

    # 7. Execution
    println("  Starting execution...")
    t_start_adapt = time()
    success = ADAPT.run!(qaoa_ansatz, trace, adapt, vqe, pool, H, ψ0, callbacks)
    t_adapt = time() - t_start_adapt
    println("  Execution completed in $(round(t_adapt, digits=2))s")

    # 8. Extract Trace Data
    final_energy = 0.0
    n_steps = 0
    if haskey(trace, :energy) && !isempty(trace[:energy])
        final_energy = trace[:energy][end]
        n_steps = length(trace[:energy])
    end

    selected_indices = Any[]
    if haskey(trace, :selected_index)
        selected_indices = trace[:selected_index]
    end

    callback_flagged = ""
    if haskey(trace, :callback_flagged)
        callback_flagged = trace[:callback_flagged]
    end

    # 9. Final State Analysis & Sampling
    t_start_sampling = time()
    final_state = ADAPT.evolve_state(qaoa_ansatz, ψ0)

    # Sampling
    samples_bitmatrix = ADAPT.sample_from_state(final_state, config.num_shots)
    # Convert BitMatrix (n_qubits x n_samples) to Vector{Vector{Bool}}
    sampled_bitstrings = [Vector{Bool}(samples_bitmatrix[:, i]) for i in 1:size(samples_bitmatrix, 2)]

    sampled_satisfactions = [get_number_of_satisfied_clauses(bs, formula) for bs in sampled_bitstrings]

    sampled_expected_satisfaction = mean(sampled_satisfactions)
    best_bs, best_sat = get_best_bitstring_among_sampled_bitstrings(sampled_bitstrings, formula)
    t_sampling = time() - t_start_sampling

    t_total = time() - t_start_total

    # Return Result Struct
    return TetrisResult(
        instance_id=instance_id,
        method="greedy",
        success=success,
        callback_flagged=callback_flagged,

        # Timing
        total_runtime=t_total,
        hamiltonian_construction_time=t_ham,
        ansatz_creation_time=t_ansatz,
        adapt_runtime=t_adapt,
        sampling_time=t_sampling, final_energy=final_energy,
        hamiltonian_terms=length(H_spv_vector),
        num_adapt_layers=length(qaoa_ansatz.γ_layers),
        num_iterations=n_steps,
        selected_indices=selected_indices,
        sampled_expected_satisfaction=sampled_expected_satisfaction,
        sampled_best_satisfaction=best_sat,
        sampled_best_solution=best_bs
    )
end
