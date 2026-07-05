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

# Names explicitly imported from dependencies/Base that are not "public" by
# ExplicitImports' measure, so the public-API import check would otherwise flag them.
const _ei_nonpublic_explicit_imports = (
    # SciMLOperators operator hierarchy: not declared `public` upstream (would need a
    # `public` declaration in a future SciMLOperators release to drop these).
    :AbstractSciMLOperator, :AbstractSciMLScalarOperator,
    # Base type-introspection internal (`Base.typename(T).wrapper`); no public equivalent
    # on any Julia version.
    :typename,
    # Public on Julia 1.11+ but only flagged on the LTS (1.10): there `ispublic` does not
    # exist, so ExplicitImports falls back to "exported", and these are public-but-not-
    # exported. Kept solely to keep the LTS QA lane green; they need no upstream change.
    :adapt_structure,            # Adapt: `public adapt_structure` (1.11+)
    :init, :solve, :solve!, :step!, # CommonSolve: `public` solver verbs (1.11+)
)
# Names explicitly imported so they are bindings in `SciMLBase` (enabling downstream
# `function SciMLBase.name(...)` method extension), but never called as a bare name
# inside SciMLBase itself, so ExplicitImports would otherwise report them as stale.
const _ei_stale_allowed = (
    # `ConstructionBase.setproperties` is extended for the solution types and accessed
    # as `SciMLBase.setproperties` by downstream packages (e.g. OrdinaryDiffEqNonlinearSolve
    # defines `function SciMLBase.setproperties(::DAEResidualJacobianWrapper, ...)`), so it
    # must be a binding in SciMLBase even though SciMLBase only ever uses the qualified form.
    :setproperties,
)
const _ei_nonpublic_qualified_accesses = (
    # --- Genuine Base/Core internals: not public on any Julia version, no public
    # equivalent exists, so the qualified access is intrinsic.
    Symbol("@max_methods"),      # Base.Experimental.@max_methods 1 (compile-time hint)
    Symbol("@pure"),             # Base.@pure (parameterless-type purity hint)
    :Experimental, :register_error_hint, # Base.Experimental error-hint registration
    :Compiler, :return_type,     # Core.Compiler.return_type (ensemble output-type inference)
    :promote_typejoin,           # Base.promote_typejoin (union-promotion in save_idxs)
    :structdiff,                 # Base.structdiff (remake NamedTuple diffing)
    :unwrap_unionall,            # Base.unwrap_unionall (type unwrapping)
    :SizeUnknown,                # Base.IteratorSize(::Type{<:DEIntegrator}) = Base.SizeUnknown()
    :filter,                     # Iterators.filter (lazy, unexported from Iterators)
    # --- Logging machinery: Base.CoreLogging internals re-exported through Logging,
    # needed to implement the custom EnsembleSolver logger; not public.
    :catch_exceptions, :handle_message, :min_enabled_level, :shouldlog,
    # --- FunctionWrappersWrappers internal error type (no public alternative).
    :NoFunctionWrapperFoundError,
    # --- Non-public dependency API: would need a `public` declaration in the
    # respective upstream release to drop these.
    :Tunable, :canonicalize, :isscimlstructure,  # SciMLStructures
    :get_discretes, :has_discretes,               # RecursiveArrayTools
    :AbstractSciMLScalarOperator,                 # SciMLOperators
    :AllVariables, :SolvedVariables,              # SymbolicIndexingInterface
    :StaticArray, :similar_type,                  # StaticArraysCore
    # --- Public on Julia 1.11+ but only flagged on the LTS (1.10), where ExplicitImports
    # falls back to "exported" because `ispublic` is unavailable. These are all
    # public-but-not-exported; kept solely to keep the LTS QA lane green, no upstream
    # change is required.
    Symbol("@propagate_inbounds"),               # Base.@propagate_inbounds (1.11+)
    :Fix1, :IteratorSize, :depwarn, :tail,        # Base (1.11+)
    :AbstractTriangular,                          # LinearAlgebra (1.11+)
    :default_rng, :seed!,                         # Random (1.11+)
    :fast_matrix_colors, :ismutable, :matrix_colors, :restructure, # ArrayInterface (1.11+)
)

run_qa(
    SciMLBase;
    aqua = false,
    explicit_imports = true,
    ei_kwargs = (;
        no_implicit_imports = (; allow_unanalyzable = _ei_unanalyzable),
        no_stale_explicit_imports = (;
            allow_unanalyzable = _ei_unanalyzable, ignore = _ei_stale_allowed,
        ),
        all_explicit_imports_are_public = (; ignore = _ei_nonpublic_explicit_imports),
        all_qualified_accesses_are_public = (; ignore = _ei_nonpublic_qualified_accesses),
    ),
)

nothing
