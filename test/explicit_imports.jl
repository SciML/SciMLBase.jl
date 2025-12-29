using ExplicitImports
using SciMLBase
using Test

@testset "ExplicitImports" begin
    # SciMLBase.ReturnCode, SciMLBase.AlgorithmInterpretation, and SciMLBase.Clocks
    # are modules created with Moshi.Data or EnumX which are not analyzable
    unanalyzable = (SciMLBase.ReturnCode, SciMLBase.AlgorithmInterpretation, SciMLBase.Clocks)

    @test check_no_implicit_imports(SciMLBase; allow_unanalyzable = unanalyzable) === nothing
    @test check_no_stale_explicit_imports(SciMLBase; allow_unanalyzable = unanalyzable) ===
          nothing
end
