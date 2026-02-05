
import Pkg
Pkg.activate(".")

include("src/MIS_TETRIS_ADAPT.jl")
import .MIS_TETRIS_ADAPT: TetrisConfig, run_greedy_tetris, get_formula_as_struct, parse_cnf_file
import ADAPT

# Load the same instance as the benchmark
cnf_path = "dataset/satqubolib/balancedsat/sat_6_vars_4.24_ratio_seed8000.cnf"
instance = parse_cnf_file(cnf_path)

config = TetrisConfig(
    adapt_type="greedy",
    initial_gamma=0.01,
    hamiltonian_type="approximate",
    num_shots=1000,
    layer_stopper_max=3,
)

println("Running greedy tetris...")
result = run_greedy_tetris(config, instance)

println("Trace parameter size: ", size(result.parameter_trace))
println("Number of layers: ", result.num_adapt_layers)
println("Number of iterations: ", result.num_iterations)

# Print the parameter trace for inspection
println("Parameter trace:")
display(result.parameter_trace)
