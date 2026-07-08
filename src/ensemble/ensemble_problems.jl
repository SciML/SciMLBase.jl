"""
$(TYPEDEF)

Container for a template problem and the user hooks used to run an ensemble of
related SciML solves.

An `EnsembleProblem` is solved by repeatedly calling `prob_func(prob, ctx)` to
construct the trajectory-specific problem, solving that problem with the
requested numerical algorithm, passing the result through `output_func(sol, ctx)`,
and combining batches with `reduction(u, data, I)`. The `ctx` argument is an
[`EnsembleContext`](@ref) that identifies the trajectory and carries
per-trajectory RNG state when `rng` or `seed` is supplied to `solve`.

## Constructor

```julia
EnsembleProblem(prob::AbstractSciMLProblem;
    output_func = (sol, ctx) -> (sol, false),
    prob_func = (prob, ctx) -> prob,
    reduction = (u, data, I) -> (append!(u, data), false),
    u_init = [], safetycopy = prob_func !== DEFAULT_PROB_FUNC)
```

## Positional Arguments

  - `prob`: The canonical problem used as the template for each trajectory.

## Keyword Arguments

  - `prob_func`: A function `(prob, ctx)` that modifies the problem for each trajectory.
    `ctx` is an [`EnsembleContext`](@ref) providing `ctx.sim_id` (unique id `1:trajectories`),
    `ctx.repeat` (rerun counter, starts at `1`), `ctx.rng` (per-trajectory RNG or `nothing`),
    `ctx.sim_seed`, and `ctx.master_rng`. `prob_func` must preserve the problem type
    of `prob`; for example, a `JumpProblem` must remain a `JumpProblem`, an
    `ODEProblem` must remain an `ODEProblem`.
  - `output_func`: A function `(sol, ctx)` that determines what is saved from each
    trajectory. It returns `(out, rerun)`, where `out` is stored in the batch output
    and `rerun` requests that the same trajectory be rerun with `ctx.repeat`
    incremented.
  - `reduction`: A function `(u, data, I)` that combines the current accumulator
    `u` with the outputs `data` from the trajectory index range `I`. It returns
    `(new_u, converged)`, where `converged=true` stops the ensemble early.
  - `u_init`: The initial accumulator passed to `reduction`. When `nothing`, the
    accumulator is initialized from the first batch output.
  - `safetycopy`: Determines whether a safety `deepcopy` is called on the `prob`
    before the `prob_func`. By default, this is true for any user-given `prob_func`,
    as without this, modifying the arguments of something in the `prob_func`, such
    as parameters or caches stored within the user function, are not necessarily
    thread-safe. If you know that your function is thread-safe, then setting this
    to `false` can improve performance when used with threads. For nested problems,
    e.g., SDE problems with custom noise processes, `deepcopy` might be
    insufficient. In such cases, use a custom `prob_func`.

## Example

```julia
function prob_func(prob, ctx)
    remake(prob, u0 = randn(ctx.rng, length(prob.u0)))
end

output_func(sol, ctx) = (sol[end, 2], false)
ensemble_prob = EnsembleProblem(prob; prob_func, output_func)
```
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
    DEFAULT_PROB_FUNC(prob, ctx)

Default `prob_func` for [`EnsembleProblem`](@ref). It returns the template
problem unchanged for every trajectory, so all trajectory-specific variation must
come from solver randomness, callbacks, or other solve-time state.
"""
DEFAULT_PROB_FUNC(prob, ctx) = prob

"""
    DEFAULT_OUTPUT_FUNC(sol, ctx)

Default `output_func` for [`EnsembleProblem`](@ref). It stores the full solution
object from each trajectory and returns `false` for the rerun flag, meaning the
trajectory is accepted on the first completion.
"""
DEFAULT_OUTPUT_FUNC(sol, ctx) = (sol, false)

"""
    DEFAULT_REDUCTION(u, data, I)

Default ensemble reduction. It appends the current batch `data` to the
accumulator `u` and returns `false` for the convergence flag, so the ensemble
continues until all requested trajectories have been run.
"""
DEFAULT_REDUCTION(u, data, I) = append!(u, data), false

"""
$(SIGNATURES)

Construct an [`EnsembleProblem`](@ref) from a template problem and optional
trajectory, output, and reduction hooks.

User-supplied hooks are passed through `prepare_function`, and `u_init` is
normalized through `prepare_initial_state`. The default `safetycopy` is `true`
when a custom `prob_func` is supplied, since mutating shared objects inside
`prob_func` is otherwise not thread-safe in threaded ensemble modes.
"""
function EnsembleProblem(
        prob;
        prob_func = DEFAULT_PROB_FUNC,
        output_func = DEFAULT_OUTPUT_FUNC,
        reduction = DEFAULT_REDUCTION,
        u_init = nothing,
        safetycopy = prob_func !== DEFAULT_PROB_FUNC
    )
    _prob_func = prepare_function(prob_func)
    _output_func = prepare_function(output_func)
    _reduction = prepare_function(reduction)
    _u_init = prepare_initial_state(u_init)
    return EnsembleProblem(prob, _prob_func, _output_func, _reduction, _u_init, safetycopy)
end

"""
$(SIGNATURES)

Keyword-only constructor for [`EnsembleProblem`](@ref). This is equivalent to
`EnsembleProblem(prob; kwargs...)` with `prob` supplied as a keyword.
"""
function EnsembleProblem(;
        prob,
        prob_func = DEFAULT_PROB_FUNC,
        output_func = DEFAULT_OUTPUT_FUNC,
        reduction = DEFAULT_REDUCTION,
        u_init = nothing, p = nothing,
        safetycopy = prob_func !== DEFAULT_PROB_FUNC
    )
    return EnsembleProblem(prob; prob_func, output_func, reduction, u_init, safetycopy)
end

"""
$(SIGNATURES)

Deprecated constructor that builds an ensemble by selecting initial conditions
from `u0s` by trajectory index.

!!! warning

    This dispatch is deprecated. See the Parallel Ensembles Simulations Interface page.
"""
function SciMLBase.EnsembleProblem(
        prob::AbstractSciMLProblem, u0s::Vector{Vector{T}}; kwargs...
    ) where {T}
    Base.depwarn(
        "This dispatch is deprecated for the standard ensemble syntax. See the Parallel \
Ensembles Simulations Interface page for more details",
        :EnsembleProblem
    )
    prob_func = (prob, ctx) -> remake(prob, u0 = u0s[ctx.sim_id])
    return SciMLBase.EnsembleProblem(prob; prob_func, kwargs...)
end

"""
$(TYPEDEF)

Weighted ensemble problem wrapper.

`WeightedEnsembleProblem` associates an [`EnsembleProblem`](@ref) with a vector
of trajectory weights. The weights are used by weighted ensemble analysis
utilities and must match the number of underlying trajectories.

## Fields

  - `ensembleprob`: The wrapped ensemble problem.
  - `weights`: A vector of trajectory weights.
"""
struct WeightedEnsembleProblem{T1 <: AbstractEnsembleProblem, T2 <: AbstractVector} <:
    AbstractEnsembleProblem
    ensembleprob::T1
    weights::T2
end

"""
Return the properties of the wrapped ensemble problem plus `:weights`.
"""
function Base.propertynames(e::WeightedEnsembleProblem)
    return (Base.propertynames(getfield(e, :ensembleprob))..., :weights)
end

"""
Access `:weights`, `:ensembleprob`, or a delegated property of the wrapped
ensemble problem.
"""
function Base.getproperty(e::WeightedEnsembleProblem, f::Symbol)
    f === :weights && return getfield(e, :weights)
    f === :ensembleprob && return getfield(e, :ensembleprob)
    return getproperty(getfield(e, :ensembleprob), f)
end

"""
$(SIGNATURES)

Construct a [`WeightedEnsembleProblem`](@ref), asserting that the supplied
`weights` sum to one and match the number of generated problems.
"""
function WeightedEnsembleProblem(args...; weights, kwargs...)
    # TODO: allow skipping checks?
    @assert sum(weights) ≈ 1
    ep = EnsembleProblem(args...; kwargs...)
    @assert length(ep.prob) == length(weights)
    return WeightedEnsembleProblem(ep, weights)
end
