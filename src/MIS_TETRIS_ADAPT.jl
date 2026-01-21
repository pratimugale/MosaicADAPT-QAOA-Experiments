module MIS_TETRIS_ADAPT

import ADAPT

# Export core types
export TetrisConfig, TetrisResult, BruteForceResult, BenchmarkResult
export run_greedy_tetris, run_bruteforce

include("config.jl")
include("results.jl")

# Utilities
include("utils/max3sat.jl")

# Include Runners
include("qaoa/greedy_tetris.jl")
include("bruteforce/bruteforce.jl")

end # module MIS_TETRIS_ADAPT
