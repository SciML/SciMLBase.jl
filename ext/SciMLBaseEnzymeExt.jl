module SciMLBaseEnzymeExt

using SciMLBase: AbstractSensitivityAlgorithm, AbstractDEAlgorithm, AbstractSciMLProblem,
    AbstractSciMLFunction, remake
import CommonSolve
import Enzyme: EnzymeRules

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

end
