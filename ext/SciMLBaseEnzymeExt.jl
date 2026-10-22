module SciMLBaseEnzymeExt

using SciMLBase: SciMLBase, AbstractSensitivityAlgorithm, AbstractDEAlgorithm,
    AbstractSciMLProblem, AbstractSciMLFunction, remake,
    AbstractDifferentiableDiscretizationMetadata, AbstractPDESolution,
    AbstractTimeseriesSolution, pde_index_cotangent, pde_call_cotangent
import CommonSolve
import Enzyme: EnzymeRules
using Enzyme: Active, Annotation, Const, Duplicated, DuplicatedNoNeed, MixedDuplicated,
    Reverse, autodiff, make_zero

# Enzyme rules for SciMLBase abstract types
#
# Sensitivity algorithms define HOW to compute derivatives, not WHAT to differentiate.
# They should be treated as inactive (constant) during Enzyme differentiation to prevent
# errors when they are stored in problem structures that Enzyme differentiates through.
#
# This fixes issues like SciMLSensitivity.jl#1225 where passing `sensealg` to ODEProblem
# constructor would fail with Enzyme.

# All sensitivity algorithm types should be inactive for Enzyme differentiation
EnzymeRules.inactive_type(::Type{<:AbstractSensitivityAlgorithm}) = true

# Solver algorithms (`Tsit5`, `Rodas5P`, the `CompositeAlgorithm` behind
# `DefaultODEAlgorithm`, …) are pure configuration: they describe HOW to solve,
# carry no derivative-carrying data, and are never differentiated with respect
# to. Marking the whole hierarchy inactive keeps Enzyme from ever assigning them
# an active activity.
#
# This is required for correctness on Julia 1.12+, whose "inline-roots" split
# calling convention (JuliaLang/julia#55767) lowers a mixed immutable struct
# (one holding both GC-tracked pointers and inline bits) into two LLVM
# arguments: an inline-bits payload and a separate roots bundle. A nested
# polyalgorithm like `CompositeAlgorithm` is exactly such a mixed struct, so it
# gets split, whereas a leaf algorithm like `Tsit5` does not. When the split
# algorithm reaches the `solve_up` custom rule in `DiffEqBaseEnzymeExt` with the
# problem marked active (e.g. an MTK problem carrying differentiable
# `MTKParameters` under `set_runtime_activity(Reverse)`), Enzyme can mark the
# two halves with disagreeing activities and trips the
# `roots_activep (DFT_CONSTANT) != activep (DFT_DUP_ARG)` assertion
# (SciMLSensitivity.jl#1499). Forcing the algorithm inactive makes both halves
# constant, so the assertion holds. This mirrors the `inactive_type` declaration
# NonlinearSolveBase uses for `NonlinearSolvePolyAlgorithm`.
EnzymeRules.inactive_type(::Type{<:AbstractDEAlgorithm}) = true

# Solver-configuration keyword arguments to `solve`, `init`, and `solve!` on
# SciML problems are never derivative-carrying — they are scalar tolerances,
# booleans, or non-numeric configuration (`abstol`, `reltol`, `verbose`,
# `alias`, `assumptions`, `Pl`, `Pr`, `weight`, …). Without an
# `inactive_kwarg` declaration, custom Enzyme rules that downstream solver
# packages register on `CommonSolve.init` / `solve!` / `solve` trip
# `Enzyme.Compiler.NonConstantKeywordArgException` when callers reach them
# through `Enzyme.gradient(set_runtime_activity(Reverse), Const(loss), …)`.
#
# Scoping to `::AbstractSciMLProblem` as the first positional argument keeps
# the declaration narrow — it applies to SciML solver calls and not to
# non-SciML CommonSolve callers, who may legitimately want differentiable
# kwargs.
EnzymeRules.inactive_kwarg(
    ::typeof(CommonSolve.init), ::AbstractSciMLProblem, args...; kwargs...,
) = nothing
EnzymeRules.inactive_kwarg(
    ::typeof(CommonSolve.solve!), ::AbstractSciMLProblem, args...; kwargs...,
) = nothing
EnzymeRules.inactive_kwarg(
    ::typeof(CommonSolve.solve), ::AbstractSciMLProblem, args...; kwargs...,
) = nothing

# Rebuilding an `AbstractSciMLFunction` via `remake` — e.g. the in-loss
# `remake(prob; build_initializeprob = true)` rebuilds `prob.f` with fresh
# `initialization_data` at `remake.jl`'s `remake(prob.f; f, initialization_data)` — is pure
# structural reconstruction. It does not itself produce derivative-carrying output: the
# active data flows through the problem's `u0`/`p` (and, at solve time, through the live
# parameters via `update_initializeprob!`), not through the rebuilt function's stored
# `initialization_data`.
#
# But `remake(::AbstractSciMLFunction; …)` is type-unstable (it infers `Any`, even when the
# inputs are concrete and the runtime result type equals `typeof(prob.f)`), so Enzyme
# differentiates the rebuild through its dynamic `runtime_generic_rev` path. There it
# cannot statically prove the type of the deeply nested `OverrideInitData` /
# `InitializationMetadata` the function carries (e.g. for MTK-built problems) and throws
# `EnzymeNoTypeError`. Marking the reconstruction call inactive — so Enzyme runs it as a
# primal-only call and does not differentiate / type-analyze its body — avoids this while
# keeping the gradient correct. This is the Enzyme analog of the
# `OverrideInitData`/`ODENLStepData` `NoTangent` declarations in the Mooncake extension.
#
# `remake(f; kwargs...)` lowers to `Core.kwcall((; kwargs...), remake, f)`, so the rule is
# declared on `Core.kwcall`.
EnzymeRules.inactive_noinl(
    ::typeof(Core.kwcall), ::Any, ::typeof(remake), ::AbstractSciMLFunction, args...,
) = true

# =============================================================================
# PDE solution wrappers
# =============================================================================
#
# The counterparts of the PDE solution rules in SciMLBaseChainRulesCoreExt, for metadata
# that subtypes `AbstractDifferentiableDiscretizationMetadata`. The shadow of the wrapper
# `solve` builds holds the shadow of the solver's solution as `original_sol`, and
# `sol[sym]`, `sol[sym, inds...]` and `sol(args...; dv)` accumulate the cotangent the
# discretizer's hooks describe into it. The solver's Enzyme rule hands the shadow of its
# solution to the adjoint, which does not read `prob.p` from it, so the parameter part of
# an observed value would be lost; a cotangent with observed terms is an error instead.

const DifferentiablePDESolution = AbstractPDESolution{
    T, N, S, D,
} where {T, N, S, D <: AbstractDifferentiableDiscretizationMetadata}

# The metadata, which the problem and the wrapped solution carry, describes the
# discretization and is a constant of the solve.
EnzymeRules.inactive_type(::Type{<:AbstractDifferentiableDiscretizationMetadata}) = true

_shadow(x::Union{Duplicated, DuplicatedNoNeed}) = x.dval
_shadow(x::MixedDuplicated) = x.dval[]
_shadow(::Annotation) = nothing

# The shadow of a wrapper of `sol`: the shadow of `sol` as `original_sol`, and with it the
# shadows of the fields the wrapper shares with `sol`. The symbols of the PDE are constants,
# and the other fields start at zero.
function _wrapper_shadow(A, sol, dsol)
    fields = map(fieldnames(typeof(A))) do name
        x = getfield(A, name)
        if name === :original_sol
            dsol
        elseif name in (:ivs, :dvs)
            x
        elseif hasfield(typeof(sol), name) && x === getfield(sol, name)
            getfield(dsol, name)
        else
            make_zero(x)
        end
    end
    return typeof(A)(fields...)
end

# Whether a shadow holds a nonzero number, through containers and the fields of structs.
_has_gradient(x::Union{AbstractFloat, Complex{<:AbstractFloat}}) = !iszero(x)
function _has_gradient(x::AbstractArray{<:Union{AbstractFloat, Complex{<:AbstractFloat}}})
    return any(!iszero, x)
end
_has_gradient(x::AbstractArray) = any(i -> isassigned(x, i) && _has_gradient(x[i]), eachindex(x))
_has_gradient(x::AbstractDict) = any(_has_gradient, values(x))
_has_gradient(x::Union{Tuple, NamedTuple}) = any(_has_gradient, x)
_has_gradient(::Union{Number, Symbol, AbstractString, Function, Type, Module}) = false
function _has_gradient(x::T) where {T}
    isstructtype(T) || return false
    return any(i -> isdefined(x, i) && _has_gradient(getfield(x, i)), 1:fieldcount(T))
end

# `u` and `interp` are reached through the rules below, so a gradient on them means a path
# that has no rule, and it is an error rather than a zero, as in SciMLBaseChainRulesCoreExt.
# `prob` shares its shadow with `original_sol`.
function _check_unreached_fields(dA)
    for field in (:u, :interp)
        _has_gradient(getfield(dA, field)) && throw(
            ArgumentError(
                "Reverse-mode differentiation reaches a PDE solution through `sol[sym]` and `sol(args...)`; the `$field` field of the solution has no rule."
            )
        )
    end
    return nothing
end

for W in (:PDETimeSeriesSolution, :PDENoTimeSolution)
    @eval begin
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfigWidth{1}, func::Const{Type{SciMLBase.$W}},
                ::Type{RT}, sol::Annotation,
                metadata::Annotation{<:AbstractDifferentiableDiscretizationMetadata}
            ) where {RT}
            A = func.val(sol.val, metadata.val)
            shadow, tape = if EnzymeRules.needs_shadow(config)
                dsol = something(_shadow(sol), make_zero(sol.val))
                # Wrapping a wrapper is the identity.
                dA = A === sol.val ? dsol : _wrapper_shadow(A, sol.val, dsol)
                (RT <: MixedDuplicated ? Ref(dA) : dA), dA
            else
                nothing, nothing
            end
            primal = EnzymeRules.needs_primal(config) ? A : nothing
            return EnzymeRules.augmented_rule_return_type(config, RT)(primal, shadow, tape)
        end
        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfigWidth{1}, func::Const{Type{SciMLBase.$W}},
                ::Type{RT}, tape, sol::Annotation,
                metadata::Annotation{<:AbstractDifferentiableDiscretizationMetadata}
            ) where {RT}
            tape === nothing || _check_unreached_fields(tape)
            return (nothing, nothing)
        end
    end
end

# Accumulate the cotangent of `A.original_sol` that a hook describes into its shadow. A
# solution that is not active, or only at runtime with a shadow that aliases it, takes none.
function _accumulate_original_sol!(A::Annotation, c)
    c === nothing && return nothing
    dA = _shadow(A)
    dA === nothing && return nothing
    dsol = dA.original_sol
    dsol === A.val.original_sol && return nothing
    isempty(c.observed) || throw(
        ArgumentError(
            "Reverse-mode differentiation of the observed values of a PDE solution is not supported with Enzyme, since the parameter part of their cotangent has no way back to the solve."
        )
    )
    c.u === nothing && return nothing
    if A.val.original_sol isa AbstractTimeseriesSolution
        for k in eachindex(dsol.u)
            dsol.u[k] .+= c.u[k]
        end
    else
        dsol.u .+= c.u
    end
    return nothing
end

# The shadow of the result of `sol[sym, inds...]` or `sol(args...)`: none for a number, which
# is active, and otherwise the cotangent the reverse pass reads, kept as the tape.
function _result_shadow(config, ::Type{RT}, y) where {RT}
    EnzymeRules.needs_shadow(config) || return nothing
    dy = make_zero(y)
    return RT <: MixedDuplicated ? Ref(dy) : dy
end

# The cotangent of the result in the reverse pass.
function _result_cotangent(dret, tape)
    dret isa Active && return dret.val
    return tape isa Ref ? tape[] : tape
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1}, func::Const{typeof(getindex)}, ::Type{RT},
        A::Annotation{<:DifferentiablePDESolution}, sym::Annotation, inds::Annotation...
    ) where {RT}
    y = A.val[sym.val, map(i -> i.val, inds)...]
    dy = _result_shadow(config, RT, y)
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    return EnzymeRules.augmented_rule_return_type(config, RT)(primal, dy, dy)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1}, func::Const{typeof(getindex)}, dret, tape,
        A::Annotation{<:DifferentiablePDESolution}, sym::Annotation, inds::Annotation...
    )
    Δ = _result_cotangent(dret, tape)
    if Δ !== nothing
        s = sym.val
        is = map(i -> i.val, inds)
        if !isempty(is)
            # `sol[sym, inds...]` is `sol[sym][inds...]` in the interface.
            Δfield = fill!(similar(A.val[s]), 0)
            view(Δfield, is...) .+= Δ  # accumulates over a repeated index
            Δ = Δfield
        end
        _accumulate_original_sol!(A, pde_index_cotangent(A.val, s, Δ))
    end
    return (nothing, nothing, map(Returns(nothing), inds)...)
end

const PDECallArgument = Union{Number, AbstractArray, Colon}

function _check_call_kwargs(kwargs)
    issubset(keys(kwargs), (:dv,)) || throw(
        ArgumentError(
            "Reverse-mode differentiation of `sol(args...; kwargs...)` covers the `dv` keyword only, got $(keys(kwargs))."
        )
    )
    return nothing
end

# The cotangent of an argument of `sol(args...)`: returned for an active number, and
# accumulated into the shadow of an array. Any other active argument, a range for instance,
# gets it through Enzyme's derivative of `collect`. Enzyme needs the type of what a rule
# returns to follow from the annotations, whatever the hook returns, hence the assertions.
function _call_arg_cotangent!(x::Active{T}, Δ) where {T}
    Δ === nothing && return make_zero(x.val)::T
    x.val isa Number && return convert(T, Δ)::T
    return first(first(autodiff(Reverse, _dot_collect, Active, x, Const(Δ))))::T
end
_dot_collect(x, Δ) = sum(collect(x) .* Δ)
function _call_arg_cotangent!(x::Union{Duplicated, DuplicatedNoNeed}, Δ)
    Δ === nothing || x.dval === x.val || (x.dval .+= Δ)
    return nothing
end
_call_arg_cotangent!(::Const, Δ) = nothing
# An argument of mixed activity, a struct of numbers and arrays, has no way to take a
# cotangent.
function _call_arg_cotangent!(x::Annotation, Δ)
    Δ === nothing && return nothing
    throw(
        ArgumentError(
            "Reverse-mode differentiation of `sol(args...)` cannot return the cotangent of an argument of type $(typeof(x.val)) with Enzyme."
        )
    )
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1}, func::Annotation{<:DifferentiablePDESolution},
        ::Type{RT}, args::Annotation{<:PDECallArgument}...; kwargs...
    ) where {RT}
    _check_call_kwargs(kwargs)
    y = func.val(map(x -> x.val, args)...; kwargs...)
    dy = _result_shadow(config, RT, y)
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    return EnzymeRules.augmented_rule_return_type(config, RT)(primal, dy, dy)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1}, func::Annotation{<:DifferentiablePDESolution},
        dret, tape, args::Annotation{<:PDECallArgument}...; kwargs...
    )
    Δ = _result_cotangent(dret, tape)
    Δ === nothing && return map(x -> _call_arg_cotangent!(x, nothing), args)
    xs = map(x -> x.val, args)
    c, Δargs = pde_call_cotangent(func.val, xs, get(kwargs, :dv, nothing), Δ)
    _accumulate_original_sol!(func, c)
    # One cotangent for each argument, for the same reason.
    return map(_call_arg_cotangent!, args, ntuple(i -> Δargs[i], Val(length(args))))
end

end
