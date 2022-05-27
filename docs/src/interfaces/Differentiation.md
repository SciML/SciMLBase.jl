# [Automatic Differentiation and Sensitivity Algorithms (Adjoints)](@id sensealg)

Automatic differentiation control is done through the `sensealg` keyword argument.
Hooks exist in the high level interfaces for `solve` which shuttle the definitions
of automatic differentiation overloads to dispatches defined in DiffEqSensitivity.jl
(should be renamed SciMLSensitivity.jl as it expands). This is done by first entering
a top-level `solve` definition, for example:

```julia
function solve(prob::DEProblem, args...; sensealg=nothing,
  u0=nothing, p=nothing, kwargs...)
  u0 = u0 !== nothing ? u0 : prob.u0
  p = p !== nothing ? p : prob.p
  if sensealg === nothing && haskey(prob.kwargs, :sensealg)
    sensealg = prob.kwargs[:sensealg]
  end
  solve_up(prob, sensealg, u0, p, args...; kwargs...)
end
```

`solve_up` then drops down the differentiable arguments as positional arguments, which
is required for the [ChainRules.jl](https://juliadiff.org/ChainRulesCore.jl/stable/)
interface. Then the `ChainRules` overloads are written on the `solve_up` calls, like:

```julia
function ChainRulesCore.frule(::typeof(solve_up), prob,
  sensealg::Union{Nothing,AbstractSensitivityAlgorithm},
  u0, p, args...;
  kwargs...)
  _solve_forward(prob, sensealg, u0, p, args...; kwargs...)
end

function ChainRulesCore.rrule(::typeof(solve_up), prob::SciMLBase.DEProblem,
  sensealg::Union{Nothing,AbstractSensitivityAlgorithm},
  u0, p, args...;
  kwargs...)
  _solve_adjoint(prob, sensealg, u0, p, args...; kwargs...)
end
```

Default definitions then exist to throw an informative error if the sensitivity
mechanism is not added:

```julia
function _concrete_solve_adjoint(args...; kwargs...)
  error("No adjoint rules exist. Check that you added `using DiffEqSensitivity`")
end

function _concrete_solve_forward(args...; kwargs...)
  error("No sensitivity rules exist. Check that you added `using DiffEqSensitivity`")
end
```

The sensitivity mechanism is kept in a separate package because of the high dependency
and load time cost introduced by the automatic differentiation libraries. Different
choices of automatic differentiation are then selected by the `sensealg` keyword argument
in `solve`, which is made into a positional argument in the `_solve_adjoint` and other
functions in order to allow dispatch.

## SensitivityADPassThrough 

The special sensitivity algorithm `SensitivityADPassThrough` is used to ignore the
internal sensitivity dispatches and instead do automatic differentiation directly
through the solver. Generally this `sensealg` is only used internally.

## Note about ForwardDiff

ForwardDiff does not use ChainRules.jl and thus it completely ignores the special
handling.

