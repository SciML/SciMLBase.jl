@doc doc"""

Defines an integral problem.
Documentation Page: https://docs.sciml.ai/Integrals/stable/

## Mathematical Specification of an Integral Problem

Integral problems are multi-dimensional integrals defined as:

```math
\int_{lb}^{ub} f(u,p) du
```

where `p` are parameters. `u` is a `Number` or `AbstractVector`
whose geometry matches the space being integrated.
This space is bounded by the lowerbound `lb` and upperbound `ub`,
which are `Number`s or `AbstractVector`s with the same geometry as `u`.

## Problem Type

### Constructors

```
IntegralProblem(f::AbstractIntegralFunction,domain,p=NullParameters(); kwargs...)
IntegralProblem(f::AbstractIntegralFunction,lb,ub,p=NullParameters(); kwargs...)
IntegralProblem(f,domain,p=NullParameters(); nout=nothing, batch=nothing, kwargs...)
IntegralProblem(f,lb,ub,p=NullParameters(); nout=nothing, batch=nothing, kwargs...)
```

- f: the integrand, callable function `y = f(u,p)` for out-of-place (default) or an
  `IntegralFunction` or `BatchIntegralFunction` for inplace and batching optimizations.
- domain: an object representing an integration domain, i.e. the tuple `(lb, ub)`.
- lb: DEPRECATED: Either a number or vector of lower bounds.
- ub: DEPRECATED: Either a number or vector of upper bounds.
- p: The parameters associated with the problem.
- nout: DEPRECATED (see `IntegralFunction`): length of the vector output of the integrand
  (by default the integrand is assumed to be scalar)
- batch: DEPRECATED (see `BatchIntegralFunction`): number of points the integrand can
  evaluate simultaneously (by default there is no batching)
- kwargs: Keyword arguments copied to the solvers.

Additionally, we can supply iip like IntegralProblem{iip}(...) as true or false to declare at
compile time whether the integrator function is in-place.

### Fields

The fields match the names of the constructor arguments.
"""
struct IntegralProblem{isinplace, P, F, T, K} <: AbstractIntegralProblem{isinplace}
    f::F
    domain::T
    p::P
    kwargs::K
    @add_kwonly function IntegralProblem{iip}(f::AbstractIntegralFunction{iip}, domain,
            p = NullParameters(); nout = nothing, batch = nothing,
            kwargs...) where {iip}
        warn_paramtype(p)
        new{iip, typeof(p), typeof(f), typeof(domain), typeof(kwargs)}(f,
            domain, p, kwargs)
    end
end

function IntegralProblem(f::AbstractIntegralFunction,
        domain,
        p = NullParameters();
        kwargs...)
    IntegralProblem{isinplace(f)}(f, domain, p; kwargs...)
end

@deprecate IntegralProblem{iip}(f::AbstractIntegralFunction,
    lb::Union{Number, AbstractVector{<:Number}},
    ub::Union{Number, AbstractVector{<:Number}},
    p = NullParameters(); kwargs...) where {iip} IntegralProblem{iip}(
    f, (lb, ub), p; kwargs...)

function IntegralProblem(f, args...; kwargs...)
    IntegralProblem{isinplace(f, 3)}(f, args...; kwargs...)
end
function IntegralProblem{iip}(
        f, args...; nout = nothing, batch = nothing, kwargs...) where {iip}
    if nout !== nothing || batch !== nothing
        @warn "`nout` and `batch` keywords are deprecated in favor of inplace `IntegralFunction`s or `BatchIntegralFunction`s. See the updated Integrals.jl documentation for details."
    end

    g = if iip
        if batch === nothing
            output_prototype = nout === nothing ? Array{Float64, 0}(undef) :
                               Vector{Float64}(undef, nout)
            IntegralFunction(f, output_prototype)
        else
            output_prototype = nout === nothing ? Float64[] :
                               Matrix{Float64}(undef, nout, 0)
            BatchIntegralFunction(f, output_prototype, max_batch = batch)
        end
    else
        if batch === nothing
            IntegralFunction(f)
        else
            BatchIntegralFunction(f, max_batch = batch)
        end
    end
    IntegralProblem(g, args...; kwargs...)
end

function Base.getproperty(prob::IntegralProblem, name::Symbol)
    if name === :lb
        domain = getfield(prob, :domain)
        lb, ub = domain
        return lb
    elseif name === :ub
        domain = getfield(prob, :domain)
        lb, ub = domain
        return ub
    elseif name === :ps
        return ParameterIndexingProxy(prob)
    end
    return Base.getfield(prob, name)
end

struct QuadratureProblem end
@deprecate QuadratureProblem(args...; kwargs...) IntegralProblem(args...; kwargs...)

@doc doc"""

Defines a integral problem over pre-sampled data.
Documentation Page: https://docs.sciml.ai/Integrals/stable/

## Mathematical Specification of a data Integral Problem

Sampled integral problems are defined as:

```math
\sum_i w_i y_i
```
where `y_i` are sampled values of the integrand, and `w_i` are weights
assigned by a quadrature rule, which depend on sampling points `x`.

## Problem Type

### Constructors

```
SampledIntegralProblem(y::AbstractArray, x::AbstractVector; dim=ndims(y), kwargs...)
```
- y: The sampled integrand, must be a subtype of `AbstractArray`.
  It is assumed that the values of `y` along dimension `dim`
  correspond to the integrand evaluated at sampling points `x`
- x: Sampling points, must be a subtype of `AbstractVector`.
- dim: Dimension along which to integrate. Defaults to the last dimension of `y`.
- kwargs: Keyword arguments copied to the solvers.

### Fields

The fields match the names of the constructor arguments.
"""
struct SampledIntegralProblem{Y, X, K} <: AbstractIntegralProblem{false}
    y::Y
    x::X
    dim::Int
    kwargs::K
    @add_kwonly function SampledIntegralProblem(y::AbstractArray, x::AbstractVector;
            dim = ndims(y),
            kwargs...)
        @assert dim<=ndims(y) "The integration dimension `dim` is larger than the number of dimensions of the integrand `y`"
        @assert length(x)==size(y, dim) "The integrand `y` must have the same length as the sampling points `x` along the integrated dimension."
        @assert axes(x, 1)==axes(y, dim) "The integrand `y` must obey the same indexing as the sampling points `x` along the integrated dimension."
        new{typeof(y), typeof(x), typeof(kwargs)}(y, x, dim, kwargs)
    end
end
