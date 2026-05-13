module MIS_TETRIS_ADAPT

import ADAPT

# Export core types
export TetrisConfig, TetrisResult, BruteForceResult, BenchmarkResult
export run_tetris, run_greedy_tetris, run_bruteforce, run_tetris_vqe, run_vanilla_qaoa, run_vanilla_vqe
export ClauseSatisfactionTracer, ApproxRatioStopper

include("config.jl")
include("results.jl")

# Utilities
include("utils/max3sat.jl")
include("utils/pauli_utils.jl")
include("utils/callbacks.jl")

# Exact Solvers
include("exact_solvers/utils.jl")
include("exact_solvers/gurobi_exact_hamiltonian.jl")

# Include Runners
# include("qaoa/qaoa_pools.jl")
include("qaoa/run_tetris.jl")
# include("qaoa/greedy_tetris.jl")
# include("bruteforce/bruteforce.jl")
end # module MIS_TETRIS_ADAPT
