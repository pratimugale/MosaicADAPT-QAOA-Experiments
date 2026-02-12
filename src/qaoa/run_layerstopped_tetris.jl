# Run generic tetris adapt on Max3SAT dataset
# This file does not use a floor stopper. In real world use-cases, the ground truth energy is unknown.

import ADAPT
import PauliOperators: ScaledPauliVector, Pauli, PauliSum, ScaledPauli
import ADAPT.ADAPT_QAOA: QAOAObservable
import Statistics: mean
import LinearAlgebra: norm

"""
    run_layerstopped_tetris(config::TetrisConfig, instance::Dict;
               pool_type::String="qaoa_double_pool",
               use_kamis::Bool=false,
               percent_tail_ends_removed::Float64=0.0,
               floor_energy::Float64=NaN)

Runs the TETRIS ADAPT algorithm (Greedy or KaMIS) on a single Max-3-SAT instance.
Returns a TetrisResult struct.
"""
function run_layerstopped_tetris(config::TetrisConfig, instance::Dict;
                    pool_type::String="qaoa_double_pool",
                    use_kamis::Bool=false,
                    percent_tail_ends_removed::Float64=0.0,
                    floor_energy::Float64=NaN)
    t_start_total = time()

    # Extract instance details
    instance_id = instance["instance_id"]
    n_vars = instance["variables"]
    method_name = use_kamis ? "kamis" : "greedy"
    println("Running TETRIS ($method_name, pool=$pool_type) on instance $instance_id with $(n_vars) variables")

    # 1. Parse Formula
    formula = MIS_TETRIS_ADAPT.get_formula_as_struct(instance["formula"])

    # 2. Construct Hamiltonian
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

    # 3. Create Pool
    local pool
    if pool_type == "qaoa_double_pool"
        pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_double_pool(n_vars)
    elseif pool_type == "qaoa_nondiagonal_double_pool"
        # We need to access this maybe from a different module if it's not exported by default
        # Assuming it is available in QAOApools or we might need to implement it/import it
        # Based on previous context, user implied it exists. 
        # If it throws, we check ADAPT structure.
        pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_nondiagonal_double_pool(n_vars)
    else
        error("Unknown pool_type: $pool_type")
    end

    # 4. Create Ansatz
    t_start_ansatz = time()
    qaoa_ansatz = ADAPT.ADAPT_QAOA.TetrisQAOAAnsatz(config.initial_gamma, pool, H)
    t_ansatz = time() - t_start_ansatz

    # 5. Initial State (Superposition)
    ψ0 = ones(ComplexF64, 2^n_vars) / sqrt(2^n_vars) # normalized

    # The normalization step looks redundant, but keeping it for now as it is in 
    #  the other examples like https://github.com/KarunyaShirali/ADAPT.jl/blob/6fa330f6192eabb159acce8fd58a58ef76228232/test/qaoa_tetris.jl#L81 
    ψ0 /= norm(ψ0)

    # 6. Setup ADAPT Algorithm with KaMIS support
    adapt = ADAPT.TETRIS_ADAPT.TETRISADAPT(
        config.gradient_threshold;
        use_kamis = use_kamis,
        kamis_seed = 42, # TODO: should we change this later?
        percent_tail_ends_removed = percent_tail_ends_removed
    )

    vqe = ADAPT.OptimOptimizer(:BFGS;
        g_tol=config.optimizer_tolerance,
        iterations=config.optimizer_max_iterations
    )

    trace = ADAPT.Trace()

    # 7. Configure Callbacks (Stoppers)
    callbacks = ADAPT.AbstractCallback[
        ADAPT.Callbacks.Tracer(:energy, :selected_index, :selected_score, :sum_gradients, :callback_flagged),
        ADAPT.Callbacks.ParameterTracer(),
        ADAPT.Callbacks.Printer(:energy),
        ADAPT.Callbacks.ScoreStopper(config.score_stopper_threshold),
        ADAPT.Callbacks.ParameterStopper(config.parameter_stopper_max),
        ADAPT.Callbacks.SlowStopper(config.slow_stopper_threshold, config.slow_stopper_patience),
        ADAPT.Callbacks.LayerStopper(config.layer_stopper_max), 
    ]

    # 8. Execution
    println("  Starting execution...")
    t_start_adapt = time()
    success = ADAPT.run!(qaoa_ansatz, trace, adapt, vqe, pool, H, ψ0, callbacks)
    t_adapt = time() - t_start_adapt
    println("  Execution completed in $(round(t_adapt, digits=2))s")

    # 9. Extract Results
    final_energy = 0.0
    n_steps = 0
    if haskey(trace, :energy) && !isempty(trace[:energy])
        final_energy = trace[:energy][end]
        n_steps = length(trace[:energy])
    end

    selected_indices = get(trace, :selected_index, Any[])
    selected_scores = get(trace, :selected_score, Float64[])
    callback_flagged = get(trace, :callback_flagged, "")
    parameter_trace = get(trace, :parameters, Any[])

    # Adaptation Energies
    adaptation_energies = Float64[]
    if haskey(trace, :energy) && haskey(trace, :adaptation)
        adapt_indices = trace[:adaptation]
        if length(adapt_indices) > 1
            adaptation_energies = trace[:energy][trace[:adaptation][2:end]]
        end
    end

    first_layer_gradient_sum = 0.0
    if haskey(trace, :sum_gradients) && !isempty(trace[:sum_gradients])
        first_layer_gradient_sum = trace[:sum_gradients][1]
    end

    # 10. Final Sampling
    t_start_sampling = time()
    final_state = ADAPT.evolve_state(qaoa_ansatz, ψ0)

    # Sampling
    samples_bitmatrix = ADAPT.sample_from_state(final_state, config.num_shots)
    # Convert BitMatrix (n_qubits x n_samples) to Vector{Vector{Bool}}
    sampled_bitstrings = [Vector{Bool}(samples_bitmatrix[:, i]) for i in 1:size(samples_bitmatrix, 2)]
    sampled_satisfactions = [MIS_TETRIS_ADAPT.get_number_of_satisfied_clauses(bs, formula) for bs in sampled_bitstrings]
    
    sampled_expected_satisfaction = mean(sampled_satisfactions)
    best_bs, best_sat = MIS_TETRIS_ADAPT.get_best_bitstring_among_sampled_bitstrings(sampled_bitstrings, formula)
    t_sampling = time() - t_start_sampling
    t_total = time() - t_start_total
    
    # Calculate satisfied percentage
    percent_satisfied = 0.0
    if length(formula) > 0
        percent_satisfied = sampled_expected_satisfaction / length(formula)
    end

    return TetrisResult(
        instance_id=instance_id,
        method=method_name, # "greedy" or "kamis"
        success=success,
        callback_flagged=callback_flagged,
        total_runtime=t_total,
        hamiltonian_construction_time=t_ham,
        ansatz_creation_time=t_ansatz,
        adapt_runtime=t_adapt,
        sampling_time=t_sampling, 
        final_energy=final_energy,
        hamiltonian_terms=length(H_spv_vector),
        num_adapt_layers=length(qaoa_ansatz.γ_layers),
        num_iterations=n_steps,
        selected_indices=selected_indices,
        selected_scores=selected_scores,
        gamma_values=qaoa_ansatz.γ_values,
        beta_values=qaoa_ansatz.parameters,
        sampled_expected_satisfaction=sampled_expected_satisfaction,
        sampled_best_satisfaction=best_sat,
        sampled_best_solution=best_bs,
        adaptation_energies=adaptation_energies,
        parameter_trace=parameter_trace,
        percent_satisfied_clauses=percent_satisfied,
        first_layer_gradient_sum=first_layer_gradient_sum
    )
end
