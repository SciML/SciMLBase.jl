using Test
using SciMLBase
using Aqua
using SciMLTesting
using Pkg

# yes this is horrible, we'll fix it when Pkg or Base provides a decent API
manifest = Pkg.Types.EnvCache().manifest
# these are good sentinels to test whether someone has added a heavy SciML package to the test deps
if haskey(manifest.deps, "NonlinearSolveBase") || haskey(manifest.deps, "DiffEqBase")
    error("Don't put Downstream Packages in non Downstream CI")
end

# https://github.com/JuliaArrays/FillArrays.jl/pull/163
@test isempty(detect_ambiguities(SciMLBase))

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
    !isempty(ambs) && @warn "Number of method ambiguities: $(length(ambs))"
    @test length(ambs) ≤ 8
end

@testset "Aqua tests (additional)" begin
    Aqua.test_undefined_exports(SciMLBase)
    Aqua.test_stale_deps(SciMLBase)
    Aqua.test_deps_compat(SciMLBase, check_extras = false)
    Aqua.test_project_extras(SciMLBase)
    # Aqua.test_project_toml_formatting(SciMLBase) # failing
    # Aqua.test_piracy(SciMLBase) # failing
end

# Qualified accesses to names their owner has not declared public. Each one buys
# compile-time behavior or a diagnostic that has no public equivalent, so it is
# allowed here rather than worked around in the source. Only checked on Julia 1.11+,
# where `Base.ispublic` exists.
const _ei_nonpublic_qualified_accesses = (
    # Compile-time hints and inference entry points. Dropping these costs inference,
    # not just style: `Vector{Any}` batches in the ensemble solvers, and a lost
    # `@max_methods` bound.
    Symbol("@max_methods"),                # Base.Experimental.@max_methods 1
    :Experimental,                         # Base.Experimental
    :Compiler, :return_type,               # Core.Compiler.return_type
    # Type introspection with no public equivalent on any supported version.
    :typename,                             # Base.typename(T).wrapper
    :unwrap_unionall,                      # Base.unwrap_unionall
    :promote_typejoin,                     # Base.promote_typejoin (keeps small Unions)
    :SizeUnknown,                          # Base.IteratorSize(::Type{<:DEIntegrator})
    # Error-hint registration and the wrapper-lookup error type it reports on.
    :register_error_hint,                  # Base.Experimental.register_error_hint
    :NoFunctionWrapperFoundError,          # FunctionWrappersWrappers
    # Not declared public upstream; dropping it costs the allocation-free
    # `totallength` path for static arrays.
    :StaticArray,                          # StaticArraysCore
)

# `AllObserved` is the RecursiveArrayTools symbolic-indexing selector, reexported so
# solution indexing code shares one selector rather than depending on its storage
# location.
run_qa(
    SciMLBase;
    reexports_allow = (:AllObserved,),
    ei_kwargs = (;
        all_qualified_accesses_are_public = (; ignore = _ei_nonpublic_qualified_accesses),
    ),
)

include("alloccheck.jl")

nothing
