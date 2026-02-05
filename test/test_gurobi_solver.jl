using Test
using SparseArrays
using PauliOperators
include("../src/exact_solvers/gurobi_solver.jl")
using .GurobiSolver

@testset "Gurobi Solver Tests" begin

    @testset "Pauli to QUBO Conversion" begin
        # 2-qubit system
        # H = 0.5*I + 0.5*Z1 + 1.0*Z1*Z2
        n = 2
        H = [
            ScaledPauli{n}(0.5, Pauli(n)),          # Identity
            ScaledPauli{n}(0.5, Pauli(n; Z=[1])),   # Z1
            ScaledPauli{n}(1.0, Pauli(n; Z=[1, 2])) # Z1 Z2
        ]
        
        Q, offset = pauli_to_qubo(H, n)
        
        # Expected calculation:
        # 0.5 -> offset += 0.5
        # 0.5 * Z1 -> 0.5 * (1 - 2x1) = 0.5 - 1.0x1. Offset += 0.5, Q[1,1] -= 1.0
        # 1.0 * Z1 Z2 -> 1.0 * (1 - 2x1 - 2x2 + 4x1x2) = 1 - 2x1 - 2x2 + 4x1x2.
        #   Offset += 1.0
        #   Q[1,1] -= 2.0
        #   Q[2,2] -= 2.0
        #   Q[1,2] += 2.0, Q[2,1] += 2.0 (symmetric distribution of 4.0 coeff)
        
        # Total Offset = 0.5 + 0.5 + 1.0 = 2.0
        @test offset ≈ 2.0
        
        # Total Q[1,1] = -1.0 - 2.0 = -3.0
        @test Q[1,1] ≈ -3.0
        
        # Total Q[2,2] = -2.0
        @test Q[2,2] ≈ -2.0
        
        # Total Q[1,2] + Q[2,1] = 4.0
        @test Q[1,2] + Q[2,1] ≈ 4.0
    end

    @testset "Gurobi Solve Simple" begin
        # Minimize Z1 Z2  -> x1 x2 terms.
        # H = Z1 Z2. Ground state: Z1=-1, Z2=1 (or vice versa) -> Energy -1.
        # Wait, Z1*Z2 eigenvalues are +1 (correlated) and -1 (anti-correlated).
        # We want to minimize. Min energy is -1.
        
        n = 2
        H = [ScaledPauli{n}(1.0, Pauli(n; Z=[1, 2]))]
        
        Q, offset = pauli_to_qubo(H, n)
        
        # Solve
        energy, solution = solve_qubo_gurobi(Q, offset)
        
        @test energy ≈ -1.0
        # Solution should be [0, 1] or [1, 0] corresponding to Z=1,-1
        # Z = 1 - 2x. 
        # If x=0 -> Z=1. If x=1 -> Z=-1.
        # Z1*Z2 = -1 requires one +1 and one -1. So one 0 and one 1.
        @test sum(solution) == 1
    end

end
