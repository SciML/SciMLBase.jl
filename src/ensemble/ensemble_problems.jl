"""
$(TYPEDEF)

## Constructor

```julia
EnsembleProblem(prob::AbstractSciMLProblem;
    output_func = (sol, i) -> (sol, false),
    prob_func = (prob, i, repeat) -> (prob),
    reduction = (u, data, I) -> (append!(u, data), false),
    u_init = [], safetycopy = prob_func !== DEFAULT_PROB_FUNC)
```

## Positional Arguments

  - `prob`: The canonical problem of the ensemble problem. This is the prob that is seeded
  into each `prob_func` call to be used as the one that is manipulated/changed for each
  run of the ensemble.

## Keyword Arguments

  - `output_func`: The function determines what is saved from the solution to the
    output array. Defaults to saving the solution itself. The output is
    `(out,rerun)` where `out` is the output and `rerun` is a boolean which
    designates whether to rerun.
  - `prob_func`: The function by which the problem is to be modified. `prob`
    is the problem, `i` is the unique id `1:trajectories` for the problem, and
    `repeat` is the iteration of the repeat. At first, it is `1`, but if
    `rerun` was true this will be `2`, `3`, etc. counting the number of times
    problem `i` has been repeated.
  - `reduction`: This function determines how to reduce the data in each batch.
    Defaults to appending the `data` into `u`, initialised via `u_data`, from
    the batches. `I` is a range of indices giving the trajectories corresponding
    to the batches. The second part of the output determines whether the simulation
    has converged. If `true`, the simulation will exit early. By default, this is
    always `false`.
  - `u_init`: The initial form of the object that gets updated in-place inside the
    `reduction` function.
  - `safetycopy`: Determines whether a safety `deepcopy` is called on the `prob`
    before the `prob_func`. By default, this is true for any user-given `prob_func`,
    as without this, modifying the arguments of something in the `prob_func`, such
    as parameters or caches stored within the user function, are not necessarily
    thread-safe. If you know that your function is thread-safe, then setting this
    to `false` can improve performance when used with threads. For nested problems,
    e.g., SDE problems with custom noise processes, `deepcopy` might be
    insufficient. In such cases, use a custom `prob_func`.

## `prob_func` Specification

One can specify a function `prob_func` which changes the problem. For example:

```julia
function prob_func(prob, i, repeat)
    @. prob.u0 = randn() * prob.u0
    prob
end
```

modifies the initial condition for all of the problems by a standard normal
random number (a different random number per simulation). Notice that since
problem types are immutable, it uses `.=`. Otherwise, one can just create
a new problem type:

```julia
function prob_func(prob, i, repeat)
    @. prob.u0 = u0_arr[i]
    prob
end
```

## `output_func` Specification

The `output_func` is a reduction function. Its arguments are the generated solution and the
unique index for the run. For example, if we wish to only save the 2nd coordinate
at the end of each solution, we can do:

```julia
output_func(sol, i) = (sol[end, 2], false)
```

Thus, the ensemble simulation would return as its data an array which is the
end value of the 2nd dependent variable for each of the runs.
"""
struct EnsembleProblem{T, T2, T3, T4, T5} <: AbstractEnsembleProblem
    prob::T
    prob_func::T2
    output_func::T3
    reduction::T4
    u_init::T5
    safetycopy::Bool
end

DEFAULT_PROB_FUNC(prob, i, repeat) = prob
DEFAULT_OUTPUT_FUNC(sol, i) = (sol, false)
DEFAULT_REDUCTION(u, data, I) = append!(u, data), false
DEFAULT_VECTOR_PROB_FUNC(prob, i, repeat) = prob[i]
function EnsembleProblem(prob::AbstractVector{<:AbstractSciMLProblem}; kwargs...)
    Base.depwarn("This dispatch is deprecated for the standard ensemble syntax. See the Parallel
    Ensembles Simulations Interface page for more details", :EnsembleProblem)
    invoke(EnsembleProblem,
        Tuple{Any},
        prob;
        prob_func = DEFAULT_VECTOR_PROB_FUNC,
        kwargs...)
end
function EnsembleProblem(prob;
        prob_func = DEFAULT_PROB_FUNC,
        output_func = DEFAULT_OUTPUT_FUNC,
        reduction = DEFAULT_REDUCTION,
        u_init = nothing,
        safetycopy = prob_func !== DEFAULT_PROB_FUNC)
    _prob_func = prepare_function(prob_func)
    _output_func = prepare_function(output_func)
    _reduction = prepare_function(reduction)
    _u_init = prepare_initial_state(u_init)
    EnsembleProblem(prob, _prob_func, _output_func, _reduction, _u_init, safetycopy)
end

function EnsembleProblem(; prob,
        prob_func = DEFAULT_PROB_FUNC,
        output_func = DEFAULT_OUTPUT_FUNC,
        reduction = DEFAULT_REDUCTION,
        u_init = nothing, p = nothing,
        safetycopy = prob_func !== DEFAULT_PROB_FUNC)
    EnsembleProblem(prob; prob_func, output_func, reduction, u_init, safetycopy)
end

#since NonlinearProblem might want to use this dispatch as well
function SciMLBase.EnsembleProblem(
        prob::AbstractSciMLProblem, u0s::Vector{Vector{T}}; kwargs...) where {T}
    Base.depwarn("This dispatch is deprecated for the standard ensemble syntax. See the Parallel
    Ensembles Simulations Interface page for more details", :EnsebleProblem)
    prob_func = (prob, i, repeat = nothing) -> remake(prob, u0 = u0s[i])
    return SciMLBase.EnsembleProblem(prob; prob_func, kwargs...)
end

struct WeightedEnsembleProblem{T1 <: AbstractEnsembleProblem, T2 <: AbstractVector} <:
       AbstractEnsembleProblem
    ensembleprob::T1
    weights::T2
end
function Base.propertynames(e::WeightedEnsembleProblem)
    (Base.propertynames(getfield(e, :ensembleprob))..., :weights)
end
function Base.getproperty(e::WeightedEnsembleProblem, f::Symbol)
    f === :weights && return getfield(e, :weights)
    f === :ensembleprob && return getfield(e, :ensembleprob)
    return getproperty(getfield(e, :ensembleprob), f)
end
function WeightedEnsembleProblem(args...; weights, kwargs...)
    # TODO: allow skipping checks?
    @assert sum(weights) ≈ 1
    ep = EnsembleProblem(args...; kwargs...)
    @assert length(ep.prob) == length(weights)
    WeightedEnsembleProblem(ep, weights)
end
