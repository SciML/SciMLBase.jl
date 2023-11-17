using Test
using SciMLBase
using Aqua

@testset "Aqua tests (performance)" begin
    # This tests that we don't accidentally run into
    # https://github.com/JuliaLang/julia/issues/29393
    Aqua.test_unbound_args(SciMLBase)

    # See: https://github.com/SciML/OrdinaryDiffEq.jl/issues/1750
    # Test that we're not introducing method ambiguities across deps
    ambs = Aqua.detect_ambiguities(SciMLBase; recursive = true)
    pkg_match(pkgname, pkdir::Nothing) = false
    pkg_match(pkgname, pkdir::AbstractString) = occursin(pkgname, pkdir)
    filter!(x -> pkg_match("SciMLBase", pkgdir(last(x).module)), ambs)

    # Uncomment for debugging:
    # for method_ambiguity in ambs
    #     @show method_ambiguity
    # end
    @warn "Number of method ambiguities: $(length(ambs))"
    @test length(ambs) ≤ 13
end

@testset "Aqua tests (additional)" begin
    Aqua.test_undefined_exports(SciMLBase)
    Aqua.test_stale_deps(SciMLBase)
    Aqua.test_deps_compat(SciMLBase, check_extras = false)
    Aqua.test_project_extras(SciMLBase)
    # Aqua.test_project_toml_formatting(SciMLBase) # failing
    # Aqua.test_piracy(SciMLBase) # failing
end

nothing
