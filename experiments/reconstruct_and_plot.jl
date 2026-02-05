
import Pkg
Pkg.activate(".")

import JSON
import LinearAlgebra: normalize!, dot
import Random
import Statistics: mean

# Include project modules
include(joinpath(@__DIR__, "..", "src", "MIS_TETRIS_ADAPT.jl"))
import .MIS_TETRIS_ADAPT: TetrisConfig, run_greedy_tetris, parse_cnf_file
import ADAPT
import .MIS_TETRIS_ADAPT: get_number_of_satisfied_clauses, get_formula_as_struct
import ADAPT.ADAPT_QAOA: TetrisQAOAAnsatz

# 1. Generate Instance (10 Qubits)
n_vars = 12
ratio = 4.24
seed = 1235
instance_type = "balanced"

println("Generating 10-qubit instance...")
python_script = joinpath(@__DIR__, "..", "src", "dataset", "satqubolib_max3sat.py")
venv_python = joinpath(@__DIR__, "..", "venv", "bin", "python3")
cmd = `$venv_python $python_script $n_vars $ratio 1 --seed $seed --type $instance_type`
run(cmd)

# Locate file (deterministic name based on logic)
# dataset/satqubolib/balancedsat/sat_10_vars_4.24_ratio_seed1234.cnf
dataset_dir = joinpath(@__DIR__, "..", "dataset", "satqubolib", "balancedsat")
target_file = joinpath(dataset_dir, "sat_12_vars_4.24_ratio_seed$(seed).cnf")

if !isfile(target_file)
    error("Generated file not found: $target_file")
end

println("Loading instance: $target_file")
instance = parse_cnf_file(target_file)
formula_struct = get_formula_as_struct(instance["formula"])

# 2. Run Greedy Tetris
println("Running Greedy Tetris...")
config = TetrisConfig(
    adapt_type="greedy",
    initial_gamma=0.01,
    hamiltonian_type="approximate",
    num_shots=1000,
    layer_stopper_max=30, # Allow enough layers to see a curve
    score_stopper_threshold=1e-10,
    slow_stopper_patience=100,
    slow_stopper_threshold=1e-9,
)

result = run_greedy_tetris(config, instance)

println("Execution complete. Layers: $(result.num_adapt_layers)")
println("Stop Reason: $(result.callback_flagged)")

# 3. Reconstruction Analysis
println("Starting Reconstruction Analysis...")

# Helper to compute Exact Expected Satisfaction
function compute_exact_satisfaction(ansatz, psi0, formula, n_vars)
    state = ADAPT.evolve_state(ansatz, psi0)
    # state is Complex vector of size 2^n
    expected_sat = 0.0
    best_sat_found = 0
    
    for i in 1:length(state)
        prob = abs2(state[i])
        if prob > 1e-10
            # Convert index (1-based) to bitstring (0-based int)
            # Integer value is i-1
            int_val = i - 1
            # Convert to Bool Vector (little-endian usually? Need to match `get_number_of_satisfied_clauses`)
            # In utils/max3sat.jl: `lit.neg ? !bitstring[lit.var] : bitstring[lit.var]`
            # lit.var is 1-based index into bitstring vector.
            # We need to check mapping. Typically:
            # basis state |b1 b2 ... bn> -> index.
            # In `PauliOperators`, KetBitString.
            # Let's use digits(val, base=2, pad=n) -> returns [LSB, ..., MSB] usually.
            # Need to confirm variable ordering.
            # Usually var 1 is LSB or MSB.
            # Let's assume standard ordering: state[1] is 00...0.
            
            bs = digits(Bool, int_val, base=2, pad=n_vars)
            # digits returns [b0, b1, ...].
            # If var 1 is indexed as bitstring[1], then var 1 is b0 (LSB).
            # This is standard for PauliOperators (Z on qubit 1 checks bit 1).
            
            sat = get_number_of_satisfied_clauses(bs, formula_struct)
            expected_sat += prob * sat
            if sat > best_sat_found
                best_sat_found = sat
            end
        end
    end
    return expected_sat, best_sat_found
end


# Reconstruction Data
history = Dict(
    "adaptations" => Int[],
    "adaptations" => Int[],
    "expected_sat" => Float64[],
    "max_sat" => Int[],
    "energy" => Float64[]
)

# Initial State
H_spv_vector = ADAPT.Hamiltonians.Max3SAT.get_approximate_hamiltonian(formula_struct, n_vars)
H = ADAPT.ADAPT_QAOA.QAOAObservable(H_spv_vector)
pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_double_pool(n_vars)

psi0 = ones(ComplexF64, 2^n_vars) / sqrt(2^n_vars)
normalize!(psi0) # Just to be safe

# Reconstruct Step-by-Step
# Note: result.parameter_trace has N rows (adaptations).
# result.selected_indices has L entries.
# They should match length, or trace handles 1-based?
# Let's check lengths.
n_steps = length(result.selected_indices)

println("Reconstructing $n_steps steps...")

for i in 1:n_steps
    # 1. Build Ansatz Structure up to step i
    ansatz_recon = TetrisQAOAAnsatz(config.initial_gamma, pool, H)
    
    # "Empty" ansatz has 0 params.
    # But TetrisQAOAAnsatz constructor initializes with empty arrays.
    
    # Loop to add layers
    for layer_idx in 1:i
        # Add Mixers
        mixers = result.selected_indices[layer_idx]
        
        # We need to know p_current BEFORE adding mixers to set γ_layer correctly?
        # In TetrisQAOA.jl adapt!: 
        #   p_current = length(ansatz.parameters)
        #   invoke(...) # adds mixers
        #   push!(γ_layers, 1 + p_current)
        
        # So:
        p_current = length(ansatz_recon.parameters)
        
        # Add the mixers
        for m_idx in mixers
            # Note: m_idx is index in pool
            # Add generator and parameter placeholder
            push!(ansatz_recon.generators, pool[m_idx])
            push!(ansatz_recon.parameters, 0.0)
        end
        
        # Add Cost Layer
        # Manually mimic insertlayer!
        push!(ansatz_recon.γ_values, ansatz_recon.γ0)
        push!(ansatz_recon.γ_layers, 1 + p_current)
    end
    
    # 2. Bind Parameters
    # Get trace row
    # Trace is Matrix column-major in Julia, but serialized as List of Lists (Columns).
    # Wait, in the JSON it was list of lists where outer list = parameter (column).
    # Inner list = row (adaptation).
    # So `result.parameter_trace` in Julia (after loading keys) is...
    # `result.parameter_trace` in struct is type `Any`. 
    # If it came from `trace[:parameters]` matrix, it's a Matrix{Float64}.
    # The JSON serializer handles Matrix -> List of Lists.
    # In Julia code, `result.parameter_trace` IS the Matrix.
    
    # Check if we have enough rows
    # Trace rows correspond to adaptations.
    if i > size(result.parameter_trace, 1)
        println("Warning: Trace has fewer rows than selected_indices. Skipping step $i.")
        continue
    end
    
    # Row i: `result.parameter_trace[i, :]`
    trace_row = result.parameter_trace[i, :]
    
    # Determine needed length
    n_params_needed = length(ansatz_recon.γ_values) + length(ansatz_recon.parameters)
    
    # Safety Check
    if n_params_needed > length(trace_row)
        println("Stopping reconstruction at step $i (Ghost Layer detected - parameters were planned but not optimized).")
        break
    end
    
    # Slice valid parameters
    params_for_step = trace_row[1:n_params_needed]
    
    # Bind
    ADAPT.bind!(ansatz_recon, params_for_step)
    
    # 3. Compute Metrics
    # Energy
    ψ = ADAPT.evolve_state(ansatz_recon, psi0)
    # E = real(ADAPT.expect(H, ψ)) # ADAPT.expect not found
    E = real(dot(ψ, H * ψ))
    
    # Comparison
    if i <= length(result.adaptation_energies)
        E_trace = result.adaptation_energies[i]
        diff = abs(E - E_trace)
        println("  Step $i: Rec. Energy = $E, Trace Energy = $E_trace, Diff = $diff")
    else
        println("  Step $i: Rec. Energy = $E, (No Trace Energy recorded for this step)")
    end
    
    # Expected Sat & Max Sat
    sat, max_sat = compute_exact_satisfaction(ansatz_recon, psi0, formula_struct, n_vars)
    
    println("  Step $i: Energy = $E, E[Sat] = $sat, Max[Sat] = $max_sat")
    
    push!(history["adaptations"], i)
    push!(history["expected_sat"], sat)
    push!(history["max_sat"], max_sat)
    push!(history["energy"], E)
end

# 4. Save Results
output_json = joinpath(@__DIR__, "..", "results", "reconstruction_data.json")
open(output_json, "w") do f
    JSON.print(f, history, 2)
end

println("Reconstruction data saved to $output_json")

# Save Full Results (Benchmark Style)
full_results_file = joinpath(@__DIR__, "..", "results", "reconstruction_full_results.json")
open(full_results_file, "w") do f
    JSON.print(f, result, 2)
end
println("Full results saved to $full_results_file")

# 5. Call Plotting Script
println("Plotting...")
plot_script = joinpath(@__DIR__, "plot_reconstruction.py")
run(`$venv_python $plot_script`) 

println("Number of clauses: $(length(instance["formula"]))")
println("Done.")
