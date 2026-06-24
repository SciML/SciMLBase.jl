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

# `ReturnCode` and `AlgorithmInterpretation` are `EnumX.@enumx`-generated submodules;
# their dynamically generated bodies are unanalyzable by ExplicitImports, so they are
# explicitly allowed rather than analyzed (they import nothing, so nothing is hidden).
const _ei_unanalyzable = (SciMLBase.ReturnCode, SciMLBase.AlgorithmInterpretation)

# Names accessed from dependencies/Base/Core/stdlib that are not marked public there
# (no public alternative exists), so they are explicitly imported and qualified-accessed.
const _ei_nonpublic_explicit_imports = (
    # SciMLPublic on Julia <1.11: `@public` expands to a no-op, so the macro is not
    # itself marked public on the LTS; public on 1.11+.
    Symbol("@public"),
    # SciMLOperators operator hierarchy (not declared public upstream).
    :AbstractSciMLOperator, :AbstractSciMLScalarOperator,
    # Adapt's extension point (`adapt` is public, the `adapt_structure` method is not).
    :adapt_structure,
    # CommonSolve's solver verbs are the canonical API but are not declared public upstream.
    :init, :solve, :solve!, :step!,
    # Base type-introspection internal (`Base.typename(T).wrapper`); no public equivalent.
    :typename,
)
const _ei_nonpublic_qualified_accesses = (
    # Base/Core internals (no public equivalents).
    Symbol("@max_methods"), Symbol("@propagate_inbounds"), Symbol("@pure"),
    :Callable, :Compiler, :Experimental, :Fix1, :IteratorSize, :SizeUnknown,
    :depwarn, :filter, :promote_typejoin, :register_error_hint, :return_type,
    :structdiff, :tail, :unwrap_unionall,
    # LinearAlgebra internal abstract type.
    :AbstractTriangular,
    # Logging machinery (custom EnsembleSolver logger).
    :catch_exceptions, :handle_message, :min_enabled_level, :shouldlog,
    # Random internals.
    :default_rng, :seed!,
    # ArrayInterface internals.
    :fast_matrix_colors, :ismutable, :matrix_colors, :restructure,
    # RecursiveArrayTools internals.
    :get_discretes, :has_discretes,
    # SciMLStructures internals.
    :Tunable, :canonicalize, :isscimlstructure,
    # SciMLOperators internal abstract type.
    :AbstractSciMLScalarOperator,
    # StaticArraysCore internals.
    :StaticArray, :similar_type,
    # SymbolicIndexingInterface internals.
    :AllVariables, :SolvedVariables,
    # FunctionWrappersWrappers internal error type.
    :NoFunctionWrapperFoundError,
)

run_qa(
    SciMLBase;
    aqua = false,
    explicit_imports = true,
    ei_kwargs = (;
        no_implicit_imports = (; allow_unanalyzable = _ei_unanalyzable),
        no_stale_explicit_imports = (; allow_unanalyzable = _ei_unanalyzable),
        all_explicit_imports_are_public = (; ignore = _ei_nonpublic_explicit_imports),
        all_qualified_accesses_are_public = (; ignore = _ei_nonpublic_qualified_accesses),
    ),
)

nothing
