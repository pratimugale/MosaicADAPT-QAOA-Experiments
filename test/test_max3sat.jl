using Test
using MIS_TETRIS_ADAPT

# We need to access internal utils if they are not exported
# Assuming MIS_TETRIS_ADAPT includes max3sat.jl but doesn't export everything.
# We can access them via module path if needed, or if included in Main for testing.
# For now, let's assume we test via the module.

@testset "Max3SAT Utils" begin

    @testset "Parsing (parse_cnf_file)" begin
        # Create a temporary known CNF file
        tmp_cnf = "temp_test.cnf"
        open(tmp_cnf, "w") do f
            println(f, "c This is a comment")
            println(f, "p cnf 3 2")
            println(f, "1 -2 3 0")
            println(f, "-1 2 -3 0")
        end

        try
            instance = MIS_TETRIS_ADAPT.parse_cnf_file(tmp_cnf)

            @test instance["variables"] == 3
            @test length(instance["formula"]) == 2

            # Check Clause 1: 1 -2 3
            c1 = instance["formula"][1]
            @test length(c1) == 3
            @test c1[1] == Dict("var" => 1, "neg" => false)
            @test c1[2] == Dict("var" => 2, "neg" => true)
            @test c1[3] == Dict("var" => 3, "neg" => false)

        finally
            rm(tmp_cnf, force=true)
        end
    end

    @testset "Logic Evaluation" begin
        # Manually construct a formula struct for testing
        # Clause 1: (x1 OR NOT x2 OR x3)
        l1 = MIS_TETRIS_ADAPT.ADAPT.Hamiltonians.Max3SAT.Types.Literal(1, false)
        l2 = MIS_TETRIS_ADAPT.ADAPT.Hamiltonians.Max3SAT.Types.Literal(2, true)
        l3 = MIS_TETRIS_ADAPT.ADAPT.Hamiltonians.Max3SAT.Types.Literal(3, false)
        c1 = MIS_TETRIS_ADAPT.ADAPT.Hamiltonians.Max3SAT.Types.Clause((l1, l2, l3))

        # Clause 2: (NOT x1 OR x2 OR NOT x3)
        l4 = MIS_TETRIS_ADAPT.ADAPT.Hamiltonians.Max3SAT.Types.Literal(1, true)
        l5 = MIS_TETRIS_ADAPT.ADAPT.Hamiltonians.Max3SAT.Types.Literal(2, false)
        l6 = MIS_TETRIS_ADAPT.ADAPT.Hamiltonians.Max3SAT.Types.Literal(3, true)
        c2 = MIS_TETRIS_ADAPT.ADAPT.Hamiltonians.Max3SAT.Types.Clause((l4, l5, l6))

        formula = MIS_TETRIS_ADAPT.ADAPT.Hamiltonians.Max3SAT.Types.Formula([c1, c2])

        # Test Case 1: bitstring = [T, T, T] (1-based index)
        # x1=T, x2=T, x3=T
        # C1: T OR F OR T -> True
        # C2: F OR T OR F -> True
        # Total = 2
        bs1 = [true, true, true]
        @test MIS_TETRIS_ADAPT.get_number_of_satisfied_clauses(bs1, formula) == 2

        # Test Case 2: bitstring = [F, T, F]
        # x1=F, x2=T, x3=F
        # C1: F OR F OR F -> False
        # C2: T OR T OR T -> True
        # Total = 1
        bs2 = [false, true, false]
        @test MIS_TETRIS_ADAPT.get_number_of_satisfied_clauses(bs2, formula) == 1
    end

    @testset "Sampling Selection" begin
        # Re-use formula from above
        l1 = MIS_TETRIS_ADAPT.ADAPT.Hamiltonians.Max3SAT.Types.Literal(1, false) # x1
        l2 = MIS_TETRIS_ADAPT.ADAPT.Hamiltonians.Max3SAT.Types.Literal(1, true)  # -x1
        l3 = MIS_TETRIS_ADAPT.ADAPT.Hamiltonians.Max3SAT.Types.Literal(2, false) # x2
        # Simple clauses to distinguish
        # C1: x1 v x1 v x1 (Needs x1=T)
        c1 = MIS_TETRIS_ADAPT.ADAPT.Hamiltonians.Max3SAT.Types.Clause((l1, l1, l1))
        # C2: -x1 v -x1 v -x1 (Needs x1=F)
        c2 = MIS_TETRIS_ADAPT.ADAPT.Hamiltonians.Max3SAT.Types.Clause((l2, l2, l2))

        formula = MIS_TETRIS_ADAPT.ADAPT.Hamiltonians.Max3SAT.Types.Formula([c1, c2])

        samples = [
            [true, false],  # Satisfies C1 (score 1)
            [false, false], # Satisfies C2 (score 1)
        ]

        # Both satisfy 1. It should return one of them.
        best_bs, score = MIS_TETRIS_ADAPT.get_best_bitstring_among_sampled_bitstrings(samples, formula)
        @test score == 1
        @test best_bs in samples
    end

end
