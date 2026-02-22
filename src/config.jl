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
    gradient_threshold::Float64 = 1e-8
    score_stopper_threshold::Float64 = 1e-6
    parameter_stopper_max::Int = 200
    layer_stopper_max::Int = 100

    # KaMIS Parameters
    use_kamis::Bool = false
    kamis_seed::Int = 42

    # Slow stopper (convergence check)
    slow_stopper_threshold::Float64 = 1e-3
    slow_stopper_patience::Int = 3
    floor_stopper_threshold::Float64 = 0.05 # Stop if energy is within this threshold of the actual ground state energy

    # Optimizer
    optimizer_tolerance::Float64 = 1e-6
    optimizer_max_iterations::Int = 100
    optimizer_jitter::Float64 = 0.0 # Standard deviation for Gaussian noise

    # Sampling
    num_shots::Int = 1000
end
