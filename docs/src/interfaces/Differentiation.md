# [Automatic Differentiation and Sensitivity Algorithms (Adjoints)](@id sensealg)

Automatic differentiation control is exposed through the `sensealg` keyword
argument to `solve`. A `sensealg` value is a lightweight
[`AbstractSensitivityAlgorithm`](@ref SciMLBase.AbstractSensitivityAlgorithm)
configuration object, or `nothing` to let the sensitivity package choose a
default. SciMLBase owns the common dispatch surface and fallback errors;
packages such as SciMLSensitivity.jl provide the concrete algorithms and
ChainRules definitions that compute derivatives of solver calls.

High-level `solve` methods make the differentiable inputs explicit before
dropping into the internal solve path. A representative shape is:

```julia
function solve(prob::AbstractDEProblem, args...; sensealg = nothing,
        u0 = nothing, p = nothing, kwargs...)
    u0 = u0 !== nothing ? u0 : prob.u0
    p = p !== nothing ? p : prob.p
    if sensealg === nothing && haskey(prob.kwargs, :sensealg)
        sensealg = prob.kwargs[:sensealg]
    end
    solve_up(prob, sensealg, u0, p, args...; kwargs...)
end
```

`solve_up` receives `sensealg`, `u0`, and `p` as positional arguments so
[ChainRules.jl](https://juliadiff.org/ChainRulesCore.jl/stable/) rules can
dispatch on the selected sensitivity algorithm and on the primal data being
differentiated. Sensitivity packages then overload the internal calls with rules
of the following form:

```julia
function ChainRulesCore.frule(::typeof(solve_up), prob,
        sensealg::Union{Nothing, AbstractSensitivityAlgorithm},
        u0, p, args...;
        kwargs...)
    _solve_forward(prob, sensealg, u0, p, args...; kwargs...)
end

function ChainRulesCore.rrule(::typeof(solve_up), prob::SciMLBase.AbstractDEProblem,
        sensealg::Union{Nothing, AbstractSensitivityAlgorithm},
        u0, p, args...;
        kwargs...)
    _solve_adjoint(prob, sensealg, u0, p, args...; kwargs...)
end
```

If no package has loaded compatible sensitivity rules, SciMLBase's default
definitions throw informative errors:

```julia
function _concrete_solve_adjoint(args...; kwargs...)
    throw(AdjointNotFoundError())
end

function _concrete_solve_forward(args...; kwargs...)
    throw(ForwardSensitivityNotFoundError())
end
```

These errors tell users to load SciMLSensitivity.jl. The sensitivity mechanism
lives outside SciMLBase because the automatic-differentiation stack has
substantial dependency and load-time cost.

## Solver and Algorithm Contracts

Solver authors should keep differentiable inputs visible to the SciMLBase solve
interface. In practice, `u0`, `p`, and differentiable keyword data should flow
through `solve` instead of being hidden in closures or global state. Solvers
should document which problem fields can receive derivatives and which outputs
may be differentiated, including interactions with `saveat`, `save_idxs`,
dense output, callbacks, events, stochastic noise, and mutation.

Concrete sensitivity algorithms should be small configuration objects. They
select how derivatives are computed, such as forward sensitivity equations,
adjoint sensitivity equations, second-order sensitivity propagation, shadowing
methods, or direct AD through solver operations. They should not store the
problem or solution being differentiated. Algorithm docstrings should state the
supported problem families, the differentiable quantities, the AD backends used
for local derivative products, and the behavior for unsupported solver features.

Forward sensitivity algorithms propagate tangent information with the primal
solve and are usually appropriate when the seed dimension is modest. Adjoint
algorithms implement reverse-mode rules that propagate cotangents from solution
values or objectives back to problem data. Second-order algorithms compute
Hessians, Hessian-vector products, or related second derivatives. Shadowing
algorithms target long-time statistics or trajectory quantities where direct
trajectory sensitivities are not the right object.

## Sensitivity Algorithm Interfaces

```@docs
SciMLBase.AbstractSensitivityAlgorithm
SciMLBase.AbstractOverloadingSensitivityAlgorithm
SciMLBase.AbstractForwardSensitivityAlgorithm
SciMLBase.AbstractAdjointSensitivityAlgorithm
SciMLBase.AbstractSecondOrderSensitivityAlgorithm
SciMLBase.AbstractShadowingSensitivityAlgorithm
```

## SensitivityADPassThrough

The special sensitivity algorithm `SensitivityADPassThrough` ignores the
SciMLBase sensitivity dispatches and asks the AD backend to differentiate
directly through the solver implementation. This is mostly an internal or
advanced fallback. It requires the selected solver path to be compatible with the
AD backend, and it will not use the specialized forward or adjoint rules that
SciMLSensitivity.jl provides.

## Note about ForwardDiff

ForwardDiff does not use ChainRules.jl and therefore ignores the ChainRules-based
`solve` handling described above. Direct AD through solver internals may also
require a pure Julia solver path and AD-compatible local operations.
