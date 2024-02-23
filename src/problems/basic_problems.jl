@doc doc"""

Defines a linear system problem.
Documentation Page: https://docs.sciml.ai/LinearSolve/stable/basics/LinearProblem/

## Mathematical Specification of a Linear Problem

### Concrete LinearProblem

To define a `LinearProblem`, you simply need to give the `AbstractMatrix` ``A``
and an `AbstractVector` ``b`` which defines the linear system:

```math
Au = b
```

### Matrix-Free LinearProblem

For matrix-free versions, the specification of the problem is given by an
operator `A(u,p,t)` which computes `A*u`, or in-place as `A(du,u,p,t)`. These
are specified via the `AbstractSciMLOperator` interface. For more details, see
the [SciMLBase Documentation](https://docs.sciml.ai/SciMLBase/stable/).

Note that matrix-free versions of LinearProblem definitions are not compatible
with all solvers. To check a solver for compatibility, use the function xxxxx.

## Problem Type

### Constructors

Optionally, an initial guess ``u₀`` can be supplied which is used for iterative
methods.

```julia
LinearProblem{isinplace}(A,x,p=NullParameters();u0=nothing,kwargs...)
LinearProblem(f::AbstractSciMLOperator,u0,p=NullParameters();u0=nothing,kwargs...)
```

`isinplace` optionally sets whether the function is in-place or not, i.e. whether
the solvers are allowed to mutate. By default this is true for `AbstractMatrix`,
and for `AbstractSciMLOperator`s it matches the choice of the operator definition.

Parameters are optional, and if not given, then a `NullParameters()` singleton
will be used, which will throw nice errors if you try to index non-existent
parameters. Any extra keyword arguments are passed on to the solvers.

### Fields

* `A`: The representation of the linear operator.
* `b`: The right-hand side of the linear system.
* `p`: The parameters for the problem. Defaults to `NullParameters`. Currently unused.
* `u0`: The initial condition used by iterative solvers.
* `kwargs`: The keyword arguments passed on to the solvers.
"""
struct LinearProblem{uType, isinplace, F, bType, P, K} <:
       AbstractLinearProblem{bType, isinplace}
    A::F
    b::bType
    u0::uType
    p::P
    kwargs::K
    @add_kwonly function LinearProblem{iip}(A, b, p = NullParameters(); u0 = nothing,
            kwargs...) where {iip}
        warn_paramtype(p)
        new{typeof(u0), iip, typeof(A), typeof(b), typeof(p), typeof(kwargs)}(A, b, u0, p,
            kwargs)
    end
end

function LinearProblem(A, b, args...; kwargs...)
    if A isa AbstractArray
        LinearProblem{true}(A, b, args...; kwargs...)
    elseif A isa Number
        LinearProblem{false}(A, b, args...; kwargs...)
    else
        LinearProblem{isinplace(A, 4)}(A, b, args...; kwargs...)
    end
end

TruncatedStacktraces.@truncate_stacktrace LinearProblem 1

"""
$(TYPEDEF)
"""
struct StandardNonlinearProblem end

@doc doc"""

Defines an interval nonlinear system problem.
Documentation Page: [https://docs.sciml.ai/NonlinearSolve/stable/basics/nonlinear_problem/](https://docs.sciml.ai/NonlinearSolve/stable/basics/nonlinear_problem/)

## Mathematical Specification of an Interval Nonlinear Problem

To define a Nonlinear Problem, you simply need to give the function ``f``
which defines the nonlinear system:

```math
f(t,p) = u = 0
```

along with an interval `tspan`, ``t \in [t_0,t_f]``, within which the root should be found.
`f` should be specified as `f(t,p)` (or in-place as `f(u,t,p)`), and `tspan` should be a
`Tuple{T,T} where T <: Number`.

!!! note

    The output value `u` is not required to be a scalar. When `u` is an `AbstractArray`, the
    problem is a simultaneous interval nonlinear problem where the solvers are made to give
    the first `t` for which any of the `u` hit zero. Currently, none of the solvers support
    this mode.

## Problem Type

### Constructors

```julia
IntervalNonlinearProblem(f::NonlinearFunction, tspan, p = NullParameters(); kwargs...)
IntervalNonlinearProblem{isinplace}(f, tspan, p = NullParameters(); kwargs...)
```

`isinplace` optionally sets whether the function is in-place or not. This is
determined automatically, but not inferred.

Parameters are optional, and if not given, then a `NullParameters()` singleton
will be used, which will throw nice errors if you try to index non-existent
parameters. Any extra keyword arguments are passed on to the solvers. For example,
if you set a `callback` in the problem, then that `callback` will be added in
every solve call.

### Fields

* `f`: The function in the problem.
* `tspan`: The interval in which the root is to be found.
* `p`: The parameters for the problem. Defaults to `NullParameters`.
* `kwargs`: The keyword arguments passed on to the solvers.
"""
struct IntervalNonlinearProblem{isinplace, tType, P, F, K, PT} <:
       AbstractIntervalNonlinearProblem{nothing, isinplace}
    f::F
    tspan::tType
    p::P
    problem_type::PT
    kwargs::K
    @add_kwonly function IntervalNonlinearProblem{iip}(
            f::AbstractIntervalNonlinearFunction{
                iip,
            },
            tspan,
            p = NullParameters(),
            problem_type = StandardNonlinearProblem();
            kwargs...) where {iip}
        warn_paramtype(p)
        new{iip, typeof(tspan), typeof(p), typeof(f),
            typeof(kwargs), typeof(problem_type)}(f,
            tspan,
            p,
            problem_type,
            kwargs)
    end

    """
    $(SIGNATURES)

    Define a steady state problem using the given function.
    `isinplace` optionally sets whether the function is inplace or not.
    This is determined automatically, but not inferred.
    """
    function IntervalNonlinearProblem{iip}(f, tspan, p = NullParameters()) where {iip}
        IntervalNonlinearProblem{iip}(IntervalNonlinearFunction{iip}(f), tspan, p)
    end
end
TruncatedStacktraces.@truncate_stacktrace IntervalNonlinearProblem 1 2

"""
$(SIGNATURES)

Define a nonlinear problem using an instance of
[`IntervalNonlinearFunction`](@ref IntervalNonlinearFunction).
"""
function IntervalNonlinearProblem(f::AbstractIntervalNonlinearFunction, tspan,
        p = NullParameters(); kwargs...)
    IntervalNonlinearProblem{isinplace(f)}(f, tspan, p; kwargs...)
end

function IntervalNonlinearProblem(f, tspan, p = NullParameters(); kwargs...)
    IntervalNonlinearProblem(IntervalNonlinearFunction(f), tspan, p; kwargs...)
end

@doc doc"""

Defines a nonlinear system problem.
Documentation Page: [https://docs.sciml.ai/NonlinearSolve/stable/basics/nonlinear_problem/](https://docs.sciml.ai/NonlinearSolve/stable/basics/nonlinear_problem/)

## Mathematical Specification of a Nonlinear Problem

To define a Nonlinear Problem, you simply need to give the function ``f``
which defines the nonlinear system:

```math
f(u,p) = 0
```

and an initial guess ``u₀`` of where `f(u, p) = 0`. `f` should be specified as `f(u, p)`
(or in-place as `f(du, u, p)`), and `u₀` should be an AbstractArray (or number)
whose geometry matches the desired geometry of `u`. Note that we are not limited
to numbers or vectors for `u₀`; one is allowed to provide `u₀` as arbitrary
matrices / higher-dimension tensors as well.

## Problem Type

### Constructors

```julia
NonlinearProblem(f::NonlinearFunction, u0, p = NullParameters(); kwargs...)
NonlinearProblem{isinplace}(f, u0, p = NullParameters(); kwargs...)
```

`isinplace` optionally sets whether the function is in-place or not. This is
determined automatically, but not inferred.

Parameters are optional, and if not given, then a `NullParameters()` singleton
will be used, which will throw nice errors if you try to index non-existent
parameters. Any extra keyword arguments are passed on to the solvers. For example,
if you set a `callback` in the problem, then that `callback` will be added in
every solve call.

For specifying Jacobians and mass matrices, see the
[NonlinearFunctions](@ref nonlinearfunctions) page.

### Fields

* `f`: The function in the problem.
* `u0`: The initial guess for the root.
* `p`: The parameters for the problem. Defaults to `NullParameters`.
* `kwargs`: The keyword arguments passed on to the solvers.
"""
struct NonlinearProblem{uType, isinplace, P, F, K, PT} <:
       AbstractNonlinearProblem{uType, isinplace}
    f::F
    u0::uType
    p::P
    problem_type::PT
    kwargs::K
    @add_kwonly function NonlinearProblem{iip}(f::AbstractNonlinearFunction{iip}, u0,
            p = NullParameters(),
            problem_type = StandardNonlinearProblem();
            kwargs...) where {iip}
        if haskey(kwargs, :p)
            error("`p` specified as a keyword argument `p = $(kwargs[:p])` to `NonlinearProblem`. This is not supported.")
        end
        warn_paramtype(p)
        new{typeof(u0), iip, typeof(p), typeof(f),
            typeof(kwargs), typeof(problem_type)}(f,
            u0,
            p,
            problem_type,
            kwargs)
    end

    """
    $(SIGNATURES)

    Define a steady state problem using the given function.
    `isinplace` optionally sets whether the function is inplace or not.
    This is determined automatically, but not inferred.
    """
    function NonlinearProblem{iip}(f, u0, p = NullParameters(); kwargs...) where {iip}
        NonlinearProblem{iip}(NonlinearFunction{iip}(f), u0, p; kwargs...)
    end
end

TruncatedStacktraces.@truncate_stacktrace NonlinearProblem 2 1
"""
$(SIGNATURES)

Define a nonlinear problem using an instance of
[`AbstractNonlinearFunction`](@ref AbstractNonlinearFunction).
"""
function NonlinearProblem(f::AbstractNonlinearFunction, u0, p = NullParameters(); kwargs...)
    NonlinearProblem{isinplace(f)}(f, u0, p; kwargs...)
end

function NonlinearProblem(f, u0, p = NullParameters(); kwargs...)
    NonlinearProblem(NonlinearFunction(f), u0, p; kwargs...)
end

"""
$(SIGNATURES)

Define a NonlinearProblem problem from SteadyStateProblem
"""
function NonlinearProblem(prob::AbstractNonlinearProblem)
    NonlinearProblem{isinplace(prob)}(prob.f, prob.u0, prob.p)
end

"""
$(SIGNATURES)

Define a nonlinear problem using an instance of
[`AbstractODEFunction`](@ref AbstractODEFunction). Note that
this is interpreted in the form of the steady state problem, i.e.
find the ODE's solution at time ``t = \\infty``.
"""
function NonlinearProblem(f::AbstractODEFunction, u0, p = NullParameters(); kwargs...)
    NonlinearProblem{isinplace(f)}(f, u0, p; kwargs...)
end

"""
$(SIGNATURES)

Define a nonlinear problem from a standard ODE problem. Note that
this is interpreted in the form of the steady state problem, i.e.
find the ODE's solution at time ``t = \\infty``
"""
function NonlinearProblem(prob::AbstractODEProblem)
    NonlinearProblem{isinplace(prob)}(prob.f, prob.u0, prob.p; prob.kwargs...)
end

@doc doc"""
Defines a nonlinear least squares problem.

## Mathematical Specification of a Nonlinear Least Squares Problem

To define a Nonlinear Problem, you simply need to give the function ``f`` which defines the
nonlinear system:

```math
\underset{x}{\min} \| f(x, p) \|
```

and an initial guess ``u_0`` for the minimization problem. ``f`` should be specified as
``f(u, p)`` (or in-place as ``f(du, u, p)``), and ``u_0`` should be an AbstractArray (or
number) whose geometry matches the desired geometry of ``u``. Note that we are not limited
to numbers or vectors for ``u_0``; one is allowed to provide ``u_0`` as arbitrary
matrices / higher-dimension tensors as well.

## Problem Type

### Constructors

```julia
NonlinearLeastSquaresProblem(f::NonlinearFunction, u0, p = NullParameters(); kwargs...)
NonlinearLeastSquaresProblem{isinplace}(f, u0, p = NullParameters(); kwargs...)
```

`isinplace` optionally sets whether the function is in-place or not. This is
determined automatically, but not inferred.

Parameters are optional, and if not given, then a `NullParameters()` singleton
will be used, which will throw nice errors if you try to index non-existent
parameters.

For specifying Jacobians and mass matrices, see the
[NonlinearFunctions](@ref nonlinearfunctions) page.

### Fields

* `f`: The function in the problem.
* `u0`: The initial guess for the solution.
* `p`: The parameters for the problem. Defaults to `NullParameters`.
* `kwargs`: The keyword arguments passed on to the solvers.
"""
struct NonlinearLeastSquaresProblem{uType, isinplace, P, F, K} <:
       AbstractNonlinearProblem{uType, isinplace}
    f::F
    u0::uType
    p::P
    kwargs::K

    @add_kwonly function NonlinearLeastSquaresProblem{iip}(
            f::AbstractNonlinearFunction{
                iip}, u0,
            p = NullParameters(); kwargs...) where {iip}
        warn_paramtype(p)
        return new{typeof(u0), iip, typeof(p), typeof(f), typeof(kwargs)}(f, u0, p, kwargs)
    end

    function NonlinearLeastSquaresProblem{iip}(f, u0, p = NullParameters()) where {iip}
        return NonlinearLeastSquaresProblem{iip}(NonlinearFunction{iip}(f), u0, p)
    end
end

TruncatedStacktraces.@truncate_stacktrace NonlinearLeastSquaresProblem 2 1

"""
$(SIGNATURES)

Define a nonlinear least squares problem using an instance of
[`AbstractNonlinearFunction`](@ref AbstractNonlinearFunction).
"""
function NonlinearLeastSquaresProblem(f::AbstractNonlinearFunction, u0,
        p = NullParameters(); kwargs...)
    return NonlinearLeastSquaresProblem{isinplace(f)}(f, u0, p; kwargs...)
end

function NonlinearLeastSquaresProblem(f, u0, p = NullParameters(); kwargs...)
    return NonlinearLeastSquaresProblem(NonlinearFunction(f), u0, p; kwargs...)
end

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

TruncatedStacktraces.@truncate_stacktrace IntegralProblem 1 4

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

TruncatedStacktraces.@truncate_stacktrace SampledIntegralProblem
