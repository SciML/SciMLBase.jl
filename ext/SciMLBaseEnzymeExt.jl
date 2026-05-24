module SciMLBaseEnzymeExt

using SciMLBase: AbstractSensitivityAlgorithm, AbstractSciMLProblem
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

end
