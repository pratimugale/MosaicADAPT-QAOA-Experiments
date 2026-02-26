"""
    TetrisConfig

Configuration for Tetris-ADAPT experiments.
"""
Base.@kwdef struct TetrisConfig
    # Physics parameters
    initial_gamma::Float64 = 0.01
    hamiltonian_type::String = "approximate" # "exact" or "approximate"
    energy_floor::Float64 = -Inf # Stop if energy <= this value

    # ADAPT-VQE parameters
    gradient_threshold::Float64 = 1e-3
    score_stopper_threshold::Float64 = 1e-3
    parameter_stopper_max::Int = 200
    layer_stopper_max::Int = 20

    # KaMIS Parameters
    use_kamis::Bool = false
    kamis_seed::Int = 42

    # Slow stopper (convergence check)
    slow_stopper_threshold::Float64 = 1e-3
    slow_stopper_patience::Int = 3
    floor_stopper_threshold::Float64 = 0.1 # Stop if energy is within this threshold of the actual ground state energy

    # Optimizer
    optimizer_tolerance::Float64 = 1e-3
    optimizer_max_iterations::Int = 100

    # Sampling
    num_shots::Int = 1000

    # Approximation ratio stopping criterion
    # Set this to the Gurobi percent satisfied clauses to enable the ApproxRatioStopper callback.
    # Leave as NaN (default) to disable.
    gurobi_percent_satisfied_threshold::Float64 = NaN
    approx_ratio_stopper_threshold::Float64 = 0.97
end
