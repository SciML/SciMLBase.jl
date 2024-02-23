# Skip the DiffEqBase handling

struct IncompatibleOptimizerError <: Exception
    err::String
end

function Base.showerror(io::IO, e::IncompatibleOptimizerError)
    print(io, e.err)
end

"""
```julia
solve(prob::OptimizationProblem, alg::AbstractOptimizationAlgorithm, args...; kwargs...)
```

## Keyword Arguments

The arguments to `solve` are common across all of the optimizers.
These common arguments are:

  - `maxiters` (the maximum number of iterations)
  - `maxtime` (the maximum of time the optimization runs for)
  - `abstol` (absolute tolerance in changes of the objective value)
  - `reltol` (relative tolerance  in changes of the objective value)
  - `callback` (a callback function)

If the chosen global optimizer employs a local optimization method,
a similar set of common local optimizer arguments exists.
The common local optimizer arguments are:

  - `local_method` (optimizer used for local optimization in global method)
  - `local_maxiters` (the maximum number of iterations)
  - `local_maxtime` (the maximum of time the optimization runs for)
  - `local_abstol` (absolute tolerance in changes of the objective value)
  - `local_reltol` (relative tolerance  in changes of the objective value)
  - `local_options` (NamedTuple of keyword arguments for local optimizer)

Some optimizer algorithms have special keyword arguments documented in the
solver portion of the documentation and their respective documentation.
These arguments can be passed as `kwargs...` to `solve`. Similarly, the special
keyword arguments for the `local_method` of a global optimizer are passed as a
`NamedTuple` to `local_options`.

Over time, we hope to cover more of these keyword arguments under the common interface.

If a common argument is not implemented for a optimizer, a warning will be shown.

## Callback Functions

The callback function `callback` is a function which is called after every optimizer
step. Its signature is:

```julia
callback = (state, loss_val, other_args) -> false
```

where `state` is a `OptimizationState` and stores information for the current
iteration of the solver and `loss_val` is loss/objective value. For more
information about the fields of the `state` look at the `OptimizationState`
documentation. The `other_args` can be the extra things returned from the
optimization `f`. This allows for saving values from the optimization and
using them for plotting and display without recalculating. The callback should
return a Boolean value, and the default should be `false`, such that the
optimization gets stopped if it returns `true`.

### Callback Example

Here we show an example a callback function that plots the prediction at the current value of the optimization variables.
The loss function here returns the loss and the prediction i.e. the solution of the `ODEProblem` `prob`, so we can use the prediction in the callback.

```julia
function predict(u)
    Array(solve(prob, Tsit5(), p = u))
end

function loss(u, p)
    pred = predict(u)
    sum(abs2, batch .- pred), pred
end

callback = function (state, l, pred; doplot = false) #callback function to observe training
    display(l)
    # plot current prediction against data
    if doplot
        pl = scatter(t, ode_data[1, :], label = "data")
        scatter!(pl, t, pred[1, :], label = "prediction")
        display(plot(pl))
    end
    return false
end
```
"""
function solve(prob::OptimizationProblem, alg, args...;
        kwargs...)::AbstractOptimizationSolution
    if supports_opt_cache_interface(alg)
        solve!(init(prob, alg, args...; kwargs...))
    else
        _check_opt_alg(prob, alg; kwargs...)
        __solve(prob, alg, args...; kwargs...)
    end
end

function SciMLBase.solve(
        prob::EnsembleProblem{T}, args...; kwargs...) where {T <: OptimizationProblem}
    return SciMLBase.__solve(prob, args...; kwargs...)
end

function _check_opt_alg(prob::OptimizationProblem, alg; kwargs...)
    !allowsbounds(alg) && (!isnothing(prob.lb) || !isnothing(prob.ub)) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) does not support box constraints. Either remove the `lb` or `ub` bounds passed to `OptimizationProblem` or use a different algorithm."))
    requiresbounds(alg) && isnothing(prob.lb) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) requires box constraints. Either pass `lb` and `ub` bounds to `OptimizationProblem` or use a different algorithm."))
    !allowsconstraints(alg) && !isnothing(prob.f.cons) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) does not support constraints. Either remove the `cons` function passed to `OptimizationFunction` or use a different algorithm."))
    requiresconstraints(alg) && isnothing(prob.f.cons) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) requires constraints, pass them with the `cons` kwarg in `OptimizationFunction`."))
    !allowscallback(alg) && haskey(kwargs, :callback) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) does not support callbacks, remove the `callback` keyword argument from the `solve` call."))
    return
end

const OPTIMIZER_MISSING_ERROR_MESSAGE = """
                                        Optimization algorithm not found. Either the chosen algorithm is not a valid solver
                                        choice for the `OptimizationProblem`, or the Optimization solver library is not loaded.
                                        Make sure that you have loaded an appropriate Optimization.jl solver library, for example,
                                        `solve(prob,Optim.BFGS())` requires `using OptimizationOptimJL` and
                                        `solve(prob,Adam())` requires `using OptimizationOptimisers`.

                                        For more information, see the Optimization.jl documentation: https://docs.sciml.ai/Optimization/stable/.
                                        """

struct OptimizerMissingError <: Exception
    alg::Any
end

function Base.showerror(io::IO, e::OptimizerMissingError)
    println(io, OPTIMIZER_MISSING_ERROR_MESSAGE)
    print(io, "Chosen Optimizer: ")
    print(e.alg)
end

"""
```julia
init(prob::OptimizationProblem, alg::AbstractOptimizationAlgorithm, args...; kwargs...)
```

## Keyword Arguments

The arguments to `init` are the same as to `solve` and common across all of the optimizers.
These common arguments are:

  - `maxiters` (the maximum number of iterations)
  - `maxtime` (the maximum of time the optimization runs for)
  - `abstol` (absolute tolerance in changes of the objective value)
  - `reltol` (relative tolerance  in changes of the objective value)
  - `callback` (a callback function)

Some optimizer algorithms have special keyword arguments documented in the
solver portion of the documentation and their respective documentation.
These arguments can be passed as `kwargs...` to `init`.

See also [`solve(prob::OptimizationProblem, alg, args...; kwargs...)`](@ref)
"""
function init(prob::OptimizationProblem, alg, args...; kwargs...)::AbstractOptimizationCache
    _check_opt_alg(prob::OptimizationProblem, alg; kwargs...)
    cache = __init(prob, alg, args...; prob.kwargs..., kwargs...)
    return cache
end

"""
```julia
solve!(cache::AbstractOptimizationCache)
```

Solves the given optimization cache.

See also [`init(prob::OptimizationProblem, alg, args...; kwargs...)`](@ref)
"""
function solve!(cache::AbstractOptimizationCache)::AbstractOptimizationSolution
    __solve(cache)
end

# needs to be defined for each cache
supports_opt_cache_interface(alg) = false
function __solve(cache::AbstractOptimizationCache)::AbstractOptimizationSolution end
function __init(prob::OptimizationProblem, alg, args...;
        kwargs...)::AbstractOptimizationCache
    throw(OptimizerMissingError(alg))
end

# if no cache interface is supported at least the following method has to be defined
function __solve(prob::OptimizationProblem, alg, args...; kwargs...)
    throw(OptimizerMissingError(alg))
end
