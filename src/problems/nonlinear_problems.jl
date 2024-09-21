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
mutable struct NonlinearProblem{uType, isinplace, P, F, K, PT} <:
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

function Base.setproperty!(prob::NonlinearProblem, s::Symbol, v)
    @warn "Mutation of NonlinearProblem detected. SciMLBase v2.0 has made NonlinearProblem temporarily mutable in order to allow for interfacing with EnzymeRules due to a current limitation in the rule system. This change is only intended to be temporary and NonlinearProblem will return to being a struct in a later non-breaking release. Do not rely on this behavior, use with caution."
    Base.setfield!(prob, s, v)
end

function Base.setproperty!(prob::NonlinearProblem, s::Symbol, v, order::Symbol)
    @warn "Mutation of NonlinearProblem detected. SciMLBase v2.0 has made NonlinearProblem temporarily mutable in order to allow for interfacing with EnzymeRules due to a current limitation in the rule system. This change is only intended to be temporary and NonlinearProblem will return to being a struct in a later non-breaking release. Do not rely on this behavior, use with caution."
    Base.setfield!(prob, s, v, order)
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
mutable struct SCCNonlinearProblem{uType, isinplace, P, F, K, PT} <:
               AbstractNonlinearProblem{uType, isinplace}
    f::F
    u0::uType
    p::P
    problem_type::PT
    kwargs::K
    @add_kwonly function SCCNonlinearProblem{iip}(f::Union{AbstractVector,Tuple}, 
            u0::Union{AbstractVector,Tuple},
            p = NullParameters(),
            problem_type = StandardNonlinearProblem();
            kwargs...) where {iip}
        if !(eltype(f) <: AbstractNonlinearFunction)
            f = NonlinearFunction{iip}.(f)
        end

        eltype(u0) <: AbstractVector || error("!(eltype(u0) <: AbstractVector) detected. SCC states must be vector.")
        length(f) != length(u0) && error("Number of SCCs undefined. length(f) = $(length(f)) != $(length(u0)) == length(u0)")
        
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
end

"""
$(SIGNATURES)
"""
function SCCNonlinearProblem(f::Union{AbstractVector,Tuple}, u0::Union{AbstractVector,Tuple}, p = NullParameters(); kwargs...)
    if !(eltype(f) <: AbstractNonlinearFunction)
        f = NonlinearFunction.(f)
    end
    all(isinplace,f) || all(!inplace,f) || error("All SCC Functions must match in-placeness")
    iip = isinplace(f[1])
    SCCNonlinearProblem{iip}(f, u0, p; kwargs...)
end