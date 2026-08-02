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

# ExplicitImports only checks an extension module that actually exists, and an extension
# module only comes into existence once its trigger package is loaded. Loading the
# weakdeps here is what puts `ext/` under QA at all.
#
# They are loaded *after* the two ambiguity assertions above on purpose: those measure
# ambiguities between SciMLBase and whatever else is resident in the process, so the
# set of loaded packages is part of the assertion. Keep them scoped to SciMLBase and
# its hard dependencies; loading the weakdep closure first would silently widen them
# (Distributions pulls in PDMats, ReverseDiff defines its own `*` methods, ...).
#
# Deliberately not loaded, so their extensions stay unscanned: SciMLBasePyCallExt and
# SciMLBaseRCallExt, because PyCall and RCall both need an external Python/R
# interpreter present at build time which the QA runner does not have. PythonCall is
# fine — it provisions its own CPython through CondaPkg.
using ChainRulesCore
using DifferentiationInterface
using Distributions
using Enzyme
using ForwardDiff
using FunctionProperties
using Makie
using Measurements
using MonteCarloMeasurements
using Mooncake
using PartialFunctions
using PythonCall
using ReverseDiff
using StaticArrays
using Tracker
using Zygote

# ExplicitImports silently skips an extension that fails to load: `Base.get_extension`
# returns `nothing` and every check reports a clean pass. Assert the extension modules
# exist rather than trusting a green `run_qa`.
@testset "Extensions loaded" begin
    for ext in (
            :SciMLBaseChainRulesCoreExt,
            :SciMLBaseDifferentiationInterfaceExt,
            :SciMLBaseDistributionsExt,
            :SciMLBaseEnzymeExt,
            :SciMLBaseForwardDiffExt,
            :SciMLBaseFunctionPropertiesExt,
            :SciMLBaseMakieExt,
            :SciMLBaseMeasurementsExt,
            :SciMLBaseMonteCarloMeasurementsExt,
            :SciMLBaseMooncakeExt,
            :SciMLBasePartialFunctionsExt,
            :SciMLBasePythonCallExt,
            :SciMLBaseReverseDiffExt,
            :SciMLBaseStaticArraysExt,
            :SciMLBaseTrackerExt,
            :SciMLBaseZygoteExt,
        )
        @test Base.get_extension(SciMLBase, ext) !== nothing
    end
end

# SciMLBase's own non-public names, reached from SciMLBase's own extensions in `ext/`.
# ExplicitImports decides "internal" by `Base.moduleroot`, and an extension module is
# its own root, so every `ext/` file looks like a third party reaching into SciMLBase.
# None of these cross a package boundary, and an extension is exactly where a package
# is expected to wire its internals to an optional dependency.
const _ei_own_internals = (
    :set_mooncakeoriginator_if_mooncake,
    :DualEltypeChecker, :ODENLStepData,
    :_reshape, :add_labels!, :anyeltypedual, :build_linear_solution, :checkkwargs,
    :diffeq_to_arrays, :getobserved, :handle_distribution_u0, :interpret_vars,
    :isdistribution, :isdualtype, :prepare_function, :prepare_initial_state,
    :reduce_tup, :responsible_map, :sse, :tmap, :totallength,
)

# Names the extensions must reach for that their *owners* have not declared public.
# Each group is an optional dependency's rule/tangent/plot-recipe interface: the
# extension exists precisely to hook into it, and there is no public spelling.
const _ei_nonpublic_third_party = (
    # Enzyme's activity interface (EnzymeCore.EnzymeRules) plus the `Core.kwcall`
    # entry point a keyword-argument method must be marked inactive through.
    :inactive_kwarg, :inactive_noinl, :inactive_type, :kwcall,
    # ForwardDiff's dual-number internals and the DiffResults buffer type it pairs
    # with, needed to opt AD configuration objects out of dual detection.
    :AbstractConfig, :Dual, :partials, :value, :DiffResult,
    # Base reflection used by the `@generated` dual-eltype scan.
    Symbol("@pure"), :isType,
    # Makie's recipe/conversion plumbing; `Makie.SpecApi` is the documented spec API
    # but is not marked public.
    :Automatic, :SpecApi, :automatic, :conversion_trait, :plotsym, :plottype,
    # Mooncake's tangent-type interface.
    :FData, :RData, :Tangent, :MutableTangent, :NoTangent, :tangent_type,
    # Tracker / ReverseDiff tracked-value internals.
    :TrackedArray, :TrackedReal, :track, :data,
    # Zygote / ZygoteRules cotangent-accumulation internals used by the ensemble and
    # observed-function adjoints.
    :ZygoteRuleConfig, :_pullback, :_tryreverse, :accum, :accum_param, :nt_nothing,
    :pair, :unzip,
    # PartialFunctions' partially-applied function type, and the Python `list` builtin.
    :PartialFunction, :list,
)

# Explicit imports (`using Foo: bar`) of names their owner has not declared public.
# Same two groups as above: SciMLBase's own internals seen across the extension
# module boundary, plus the AD interfaces the extensions are written against.
const _ei_nonpublic_explicit_imports = (
    _ei_own_internals...,
    # Mooncake's rule-definition interface.
    Symbol("@is_primitive"), Symbol("@mooncake_overlay"), Symbol("@zero_adjoint"),
    :CoDual, :MinimalCtx, :NoFData, :NoPullback, :NoRData, :NoTangent, :build_rrule,
    :fdata, :instantiate, :lazy_zero_rdata, :primal, :rrule!!, :tangent,
    :zero_fcodual, :zero_tangent,
    # ZygoteRules' `getproperty`/`getfield` adjoint hooks.
    :literal_getfield, :literal_getproperty,
)

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
    _ei_own_internals...,
    _ei_nonpublic_third_party...,
)

# The CommonSolve verbs. SciMLBase does not own them, but they are the SciML solve
# interface as users and solver packages write it, and they are documented here, so
# they stay exported and are allow-listed rather than dropped. They are documented at
# CommonSolve as well, so the rendered-docs check skips them.
const _reexports_allow = (:init, :solve, :solve!, :step!)

run_qa(
    SciMLBase;
    reexports_allow = _reexports_allow,
    api_docs_kwargs = (; rendered_ignore = _reexports_allow),
    ei_kwargs = (;
        all_qualified_accesses_are_public = (; ignore = _ei_nonpublic_qualified_accesses),
        all_explicit_imports_are_public = (; ignore = _ei_nonpublic_explicit_imports),
    ),
)

include("alloccheck.jl")

nothing
