
import Pkg
Pkg.activate(".")

include(joinpath(@__DIR__, "..", "src", "MIS_TETRIS_ADAPT.jl"))
import .MIS_TETRIS_ADAPT: parse_cnf_file, convert_formula_to_clauses, solve_max_e3sat_exact, get_formula_as_struct
import JSON

function filter_instances(input_dir::String, output_dir::String, rejected_dir::String)
    mkpath(output_dir)
    mkpath(rejected_dir)

    files = readdir(input_dir)
    cnf_files = filter(f -> endswith(f, ".cnf"), files)

    println("Processing $(length(cnf_files)) instances from $input_dir...")

    for f in cnf_files
        path = joinpath(input_dir, f)
        instance = parse_cnf_file(path)
        formula = get_formula_as_struct(instance["formula"])
        clauses = convert_formula_to_clauses(formula)
        n_vars = instance["variables"]

        # Solve exactly with Gurobi
        max_sat, _ = solve_max_e3sat_exact(n_vars, clauses)

        is_satisfiable = (max_sat == length(clauses))

        if is_satisfiable
            println("[SAT] $f is SAT. Moving to $output_dir")
            mv(path, joinpath(output_dir, f), force=true)
        else
            println("[UNSAT] $f is UNSAT ($max_sat/$(length(clauses))). Moving to $rejected_dir")
            mv(path, joinpath(rejected_dir, f), force=true)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 3
        println("Usage: julia filter_satisfiable.jl <input_dir> <output_dir> <rejected_dir>")
        exit(1)
    end
    filter_instances(ARGS[1], ARGS[2], ARGS[3])
end
