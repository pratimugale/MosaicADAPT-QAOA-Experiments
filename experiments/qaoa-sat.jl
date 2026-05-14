
import Pkg
Pkg.activate(".")

import JSON
import Dates
import Random
using Printf
using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "MIS_TETRIS_ADAPT.jl"))
import .MIS_TETRIS_ADAPT: TetrisConfig, TetrisResult, parse_cnf_file
import .MIS_TETRIS_ADAPT: run_tetris, run_vanilla_qaoa

function run_qaoa_benchmark(input_dir::String, output_dir::String, override_id::Union{Int, Nothing}=nothing)
    # Explicitly set single threading for BLAS
    LinearAlgebra.BLAS.set_num_threads(1)
    
    mkpath(output_dir)
    if isfile(input_dir)
        cnf_files = [basename(input_dir)]
        input_dir = dirname(input_dir)
    else
        cnf_files = filter(f -> endswith(f, ".cnf"), readdir(input_dir))
    end
    
    println("Benchmarking $(length(cnf_files)) instances from $input_dir...")

    # Precision
    tol = 1e-6

    gammas = [0.01]

    # 5 protocols:
    # 1. Adapt-QAOA (Vanilla) - one by one
    # 2. Tetris-QAOA (Greedy)
    # 3. Tetris-QAOA (KaMIS)
    # 4. Regular QAOA (Standard Mixer, Greedy selection)
    # 5. Regular QAOA (Standard Mixer, KaMIS selection - though redundant)

    for (idx, cnf_file) in enumerate(cnf_files)
        # Use override ID if provided (useful for Slurm Array Jobs)
        actual_id = (length(cnf_files) == 1 && !isnothing(override_id)) ? override_id : idx
        
        println("\n>>> Processing instance $idx/$(length(cnf_files)) (ID: $actual_id): $cnf_file")
        instance = parse_cnf_file(joinpath(input_dir, cnf_file))
        instance["instance_id"] = actual_id

        for gamma in gammas
            config = TetrisConfig(
                initial_gamma=gamma,
                optimizer_tolerance=tol,
                gradient_threshold=tol,
                score_stopper_threshold=tol,
                slow_stopper_threshold=tol,
                layer_stopper_max=20,
                num_shots=1000,
                hamiltonian_type="exact"
            )

            # 1. Adapt-QAOA
            println("  Running Adapt-QAOA (gamma=$gamma)")
            res_vanilla = run_vanilla_qaoa(config, instance, pool_type="qaoa_nondiagonal_double_pool", instance_filename=cnf_file)
            save_result(res_vanilla, output_dir, "adapt_qaoa", gamma)

            # 2. Tetris-QAOA
            println("  Running Tetris-QAOA Greedy (gamma=$gamma)")
            res_greedy = run_tetris(config, instance, pool_type="qaoa_nondiagonal_double_pool", use_kamis=false, method_name="tetris_qaoa_greedy", instance_filename=cnf_file)
            save_result(res_greedy, output_dir, "tetris_qaoa", gamma)

            # 3. MosaicADAPT-QAOA
            println("  Running MosaicADAPT-QAOA (gamma=$gamma)")
            res_kamis = run_tetris(config, instance, pool_type="qaoa_nondiagonal_double_pool", use_kamis=true, method_name="mosaic_adapt_qaoa", instance_filename=cnf_file)
            save_result(res_kamis, output_dir, "mosaicadapt_qaoa", gamma)
        end
    end
end

function save_result(res::TetrisResult, dir::String, type::String, gamma::Float64)
    # Use instance filename for uniqueness in parallel runs if it exists
    inst_label = isempty(res.instance_filename) ? "inst$(res.instance_id)" : replace(basename(res.instance_filename), ".cnf" => "")
    filename = "qaoa_$(type)_gamma$(gamma)_$(inst_label).json"
    open(joinpath(dir, filename), "w") do f
        JSON.print(f, res, 2)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println("Usage: julia qaoa-sat.jl <input_dir_or_file> <output_dir> [override_instance_id]")
        exit(1)
    end
    
    input_path = ARGS[1]
    output_dir = ARGS[2]
    override_id = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : nothing

    run_qaoa_benchmark(input_path, output_dir, override_id)
end
