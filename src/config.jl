"""
    TetrisConfig

Configuration for Tetris-ADAPT experiments.
"""
Base.@kwdef struct TetrisConfig
    # Method ("greedy" or "mis")
    adapt_type::String = "greedy"

    # Physics parameters
    initial_gamma::Float64 = 0.01
    hamiltonian_type::String = "approximate" # "exact" or "approximate"
    energy_floor::Float64 = -Inf # Stop if energy <= this value

    # ADAPT-VQE parameters
    gradient_threshold::Float64 = 1e-3 # this is a filter - remove operators with gradient < this
    score_stopper_threshold::Float64 = 1e-3 # this is a termination condition - stop if score < this
    parameter_stopper_max::Int = 1000 # maximum number of parameters to keep
    layer_stopper_max::Int = 3 # maximum number of QAOA layers (p)

    # Slow stopper (convergence check)
    slow_stopper_threshold::Float64 = 1e-2
    slow_stopper_patience::Int = 3
    floor_stopper_threshold::Float64 = 0.05 # Threshold for FloorStopper

    # Optimizer
    optimizer_tolerance::Float64 = 1e-2
    optimizer_max_iterations::Int = 100

    # Sampling
    num_shots::Int = 1000
end
