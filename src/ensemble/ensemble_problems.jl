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
  - `reduction`: This function is used to aggregate the results in each simulation batch. By default, it appends the `data` from the batch to `u`, which is initialized via `u_data`. The `I` is a range of indices corresponding to the trajectories for the current batch.

### Arguments:
- `u`: The solution from the current ensemble run. This is the accumulated data that gets updated in each batch.
- `data`: The results from the current batch of simulations. This is typically some data (e.g., variable values, time steps) that is merged with `u`.
- `I`: A range of indices corresponding to the simulations in the current batch. This provides the trajectory indices for the batch.

### Returns:
- `(new_data, has_converged)`: A tuple where:
  - `new_data`: The updated accumulated data, typically the result of appending `data` to `u`.
  - `has_converged`: A boolean indicating whether the simulation has converged and should terminate early. If `true`, the simulation will stop early. If `false`, the simulation will continue. By default, this is `false`, meaning the simulation will not stop early.

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

# Defines a structure to manage an ensemble (batch) of problems.
# Each field controls how the ensemble behaves during simulation.

struct EnsembleProblem{T, T2, T3, T4, T5} <: AbstractEnsembleProblem
    prob::T         # The original base problem to replicate or modify.
    prob_func::T2   # A function defining how to generate each subproblem (e.g., changing initial conditions).
    output_func::T3 # A function to post-process each individual simulation result.
    reduction::T4 # A function to combine results from all simulations.
    u_init::T5 # The initial container used to accumulate the results.
    safetycopy::Bool # Whether to copy the problem when creating subproblems (to avoid unintended modifications).
end

# Returns the same problem without modification.
DEFAULT_PROB_FUNC(prob, i, repeat) = prob

# Returns the solution as-is, along with a flag (false) indicating no early termination.
DEFAULT_OUTPUT_FUNC(sol, i) = (sol, false)

# Appends new data to the accumulated data, no early convergence.
DEFAULT_REDUCTION(u, data, I) = append!(u, data), false

# Selects the i-th problem from a vector of problems.
DEFAULT_VECTOR_PROB_FUNC(prob, i, repeat) = prob[i]

# Constructor: creates an EnsembleProblem when the input is a vector of problems (DEPRECATED).
function EnsembleProblem(prob::AbstractVector{<:AbstractSciMLProblem}; kwargs...)
    Base.depwarn("This dispatch is deprecated for the standard ensemble syntax. See the Parallel
    Ensembles Simulations Interface page for more details", :EnsembleProblem)
    invoke(EnsembleProblem,
        Tuple{Any},
        prob;
        prob_func = DEFAULT_VECTOR_PROB_FUNC,
        kwargs...)
end

# Main constructor: creates an EnsembleProblem with optional custom behavior.
function EnsembleProblem(prob;
        prob_func = DEFAULT_PROB_FUNC,
        output_func = DEFAULT_OUTPUT_FUNC,
        reduction = DEFAULT_REDUCTIO"""
$(TYPEDEF)

Defines a structure to manage an ensemble (batch) of problems.
Each field controls how the ensemble behaves during simulation.

## Arguments:

- `prob`: The original base problem to replicate or modify.
- `prob_func`: A function that defines how to generate each subproblem (e.g., changing initial conditions).
- `output_func`: A function to post-process each individual simulation result.
- `reduction`: A function to combine results from all simulations.
- `u_init`: The initial container used to accumulate the results.
- `safetycopy`: Whether to copy the problem when creating subproblems (to avoid unintended modifications).
"""
struct EnsembleProblem{T, T2, T3, T4, T5} <: AbstractEnsembleProblem
    prob::T
    prob_func::T2
    output_func::T3
    reduction::T4
    u_init::T5
    safetycopy::Bool
end

"""
Returns the same problem without modification.

"""
DEFAULT_PROB_FUNC(prob, i, repeat) = prob

"""
Returns the solution as-is, along with `false` indicating no rerun.

"""
DEFAULT_OUTPUT_FUNC(sol, i) = (sol, false)

"""
Appends new data to the accumulated data and returns `false` to indicate no early termination.

"""
DEFAULT_REDUCTION(u, data, I) = append!(u, data), false

"""
Selects the i-th problem from a vector of problems.

"""
DEFAULT_VECTOR_PROB_FUNC(prob, i, repeat) = prob[i]

"""
$(TYPEDEF)

Constructor that creates an EnsembleProblem when the input is a vector of problems.

!!! warning
    This constructor is deprecated. Use the standard ensemble syntax with `prob_func` instead.
"""
function EnsembleProblem(prob::AbstractVector{<:AbstractSciMLProblem}; kwargs...)
    Base.depwarn("This dispatch is deprecated for the standard ensemble syntax. See the Parallel Ensembles Simulations Interface page for more details", :EnsembleProblem)
    invoke(EnsembleProblem,
        Tuple{Any},
        prob;
        prob_func = DEFAULT_VECTOR_PROB_FUNC,
        kwargs...)
end

"""
$(TYPEDEF)

Main constructor for `EnsembleProblem`.

## Arguments:

- `prob`: The base problem.
- `prob_func`: Function to modify the base problem per trajectory.
- `output_func`: Function to extract output from a solution.
- `reduction`: Function to aggregate results.
- `u_init`: Initial value for aggregation.
- `safetycopy`: Whether to deepcopy the problem before modifying.

"""
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

"""
$(TYPEDEF)

Alternate constructor that uses only keyword arguments.

"""
function EnsembleProblem(; prob,
        prob_func = DEFAULT_PROB_FUNC,
        output_func = DEFAULT_OUTPUT_FUNC,
        reduction = DEFAULT_REDUCTION,
        u_init = nothing, p = nothing,
        safetycopy = prob_func !== DEFAULT_PROB_FUNC)
    EnsembleProblem(prob; prob_func, output_func, reduction, u_init, safetycopy)
end

"""
$(TYPEDEF)

Constructor for NonlinearProblem.

!!! warning
    This dispatch is deprecated. See the Parallel Ensembles Simulations Interface page.

"""
function SciMLBase.EnsembleProblem(
        prob::AbstractSciMLProblem, u0s::Vector{Vector{T}}; kwargs...) where {T}
    Base.depwarn("This dispatch is deprecated for the standard ensemble syntax. See the Parallel Ensembles Simulations Interface page for more details", :EnsembleProblem)
    prob_func = (prob, i, repeat = nothing) -> remake(prob, u0 = u0s[i])
    return SciMLBase.EnsembleProblem(prob; prob_func, kwargs...)
end

"""
$(TYPEDEF)

Defines a weighted version of an `EnsembleProblem`, where different simulations contribute unequally.

## Arguments:

- `ensembleprob`: The base ensemble problem.
- `weights`: A vector of weights corresponding to each simulation.

"""
struct WeightedEnsembleProblem{T1 <: AbstractEnsembleProblem, T2 <: AbstractVector} <:
       AbstractEnsembleProblem
    ensembleprob::T1
    weights::T2
end

"""

Returns a list of all accessible properties, including those from the inner ensemble and `:weights`.

"""
function Base.propertynames(e::WeightedEnsembleProblem)
    (Base.propertynames(getfield(e, :ensembleprob))..., :weights)
end

"""
Accesses properties of a `WeightedEnsembleProblem`.

Returns `weights` or delegates to the underlying ensemble.

"""
function Base.getproperty(e::WeightedEnsembleProblem, f::Symbol)
    f === :weights && return getfield(e, :weights)
    f === :ensembleprob && return getfield(e, :ensembleprob)
    return getproperty(getfield(e, :ensembleprob), f)
end

"""
$(TYPEDEF)

Constructor for `WeightedEnsembleProblem`. Ensures weights sum to 1 and matches problem count.

"""
function WeightedEnsembleProblem(args...; weights, kwargs...)
    @assert sum(weights) ≈ 1
    ep = EnsembleProblem(args...; kwargs...)
    @assert length(ep.prob) == length(weights)
    WeightedEnsembleProblem(ep, weights)
end
N,
        u_init = nothing,
        safetycopy = prob_func !== DEFAULT_PROB_FUNC)
    _prob_func = prepare_function(prob_func)
    _output_func = prepare_function(output_func)
    _reduction = prepare_function(reduction)
    _u_init = prepare_initial_state(u_init)
    EnsembleProblem(prob, _prob_func, _output_func, _reduction, _u_init, safetycopy)
end

# Alternative constructor that accepts parameters through keyword arguments (especially used internally).
function EnsembleProblem(; prob,
        prob_func = DEFAULT_PROB_FUNC,
        output_func = DEFAULT_OUTPUT_FUNC,
        reduction = DEFAULT_REDUCTION,
        u_init = nothing, p = nothing,
        safetycopy = prob_func !== DEFAULT_PROB_FUNC)
    EnsembleProblem(prob; prob_func, output_func, reduction, u_init, safetycopy)
end

#since NonlinearProblem might want to use this dispatch as well
#Special constructor used for creating an EnsembleProblem where initial states vary.
function SciMLBase.EnsembleProblem(
        prob::AbstractSciMLProblem, u0s::Vector{Vector{T}}; kwargs...) where {T}
    Base.depwarn("This dispatch is deprecated for the standard ensemble syntax. See the Parallel
    Ensembles Simulations Interface page for more details", :EnsebleProblem)
    prob_func = (prob, i, repeat = nothing) -> remake(prob, u0 = u0s[i])
    return SciMLBase.EnsembleProblem(prob; prob_func, kwargs...)
end

# Defines a weighted version of an EnsembleProblem, where different simulations contribute unequally.
struct WeightedEnsembleProblem{T1 <: AbstractEnsembleProblem, T2 <: AbstractVector} <:
       AbstractEnsembleProblem
    ensembleprob::T1 # The base ensemble problem.
    weights::T2  # A vector of weights corresponding to each simulation.
end

# Allow accessing all properties from the base ensemble plus the new weights field.
function Base.propertynames(e::WeightedEnsembleProblem)
    (Base.propertynames(getfield(e, :ensembleprob))..., :weights)
end

# Getter for fields: either return weights, ensembleprob, or delegate to the underlying ensemble.
function Base.getproperty(e::WeightedEnsembleProblem, f::Symbol)
    f === :weights && return getfield(e, :weights)
    f === :ensembleprob && return getfield(e, :ensembleprob)
    return getproperty(getfield(e, :ensembleprob), f)
end

# Constructor for WeightedEnsembleProblem, checks that weights sum to ~1.
function WeightedEnsembleProblem(args...; weights, kwargs...)
    # TODO: allow skipping checks?
    @assert sum(weights) ≈ 1
    ep = EnsembleProblem(args...; kwargs...)
    @assert length(ep.prob) == length(weights)
    WeightedEnsembleProblem(ep, weights)
end
