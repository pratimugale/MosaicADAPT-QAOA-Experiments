module MIS_TETRIS_ADAPT

import ADAPT

# Export core types
export TetrisConfig, TetrisResult, BruteForceResult, BenchmarkResult
export run_greedy_tetris, run_bruteforce

include("config.jl")
include("results.jl")

# Utilities
include("utils/max3sat.jl")
include("utils/callbacks.jl")

# Exact Solvers
include("exact_solvers/utils.jl")
include("exact_solvers/gurobi_exact_hamiltonian.jl")

# Include Runners
include("qaoa/qaoa_pools.jl")
include("qaoa/greedy_tetris.jl")
include("bruteforce/bruteforce.jl")
include("qaoa/run_layerstopped_tetris.jl")
end # module MIS_TETRIS_ADAPT
