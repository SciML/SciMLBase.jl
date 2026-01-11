using ExplicitImports
using SciMLBase
using Test

@testset "ExplicitImports" begin
    @testset "No implicit imports" begin
        @test check_no_implicit_imports(
            SciMLBase;
            allow_unanalyzable = (
                SciMLBase.ReturnCode,
                SciMLBase.AlgorithmInterpretation,
                SciMLBase.Clocks
            )
        ) === nothing
    end

    @testset "No stale explicit imports" begin
        @test check_no_stale_explicit_imports(
            SciMLBase;
            allow_unanalyzable = (
                SciMLBase.ReturnCode,
                SciMLBase.AlgorithmInterpretation,
                SciMLBase.Clocks
            )
        ) === nothing
    end
end
