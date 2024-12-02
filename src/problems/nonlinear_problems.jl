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
    SCCNonlinearProblem(probs, explicitfuns!)

Defines an SCC-split nonlinear system to be solved iteratively.

## Mathematical Specification of an SCC-Split Nonlinear Problem

An SCC-Split Nonlinear Problem is a system of nonlinear equations

```math
f(u,p) = 0
```

with the special property that its Jacobian is in block-lower-triangular
form. In this form, the nonlinear problem can be decomposed into a system
of nonlinear systems. 

```math
f_1(u_1,p) = 0
f_2(u_2,u_1,p) = 0
f_3(u_3,u_2,u_1,p) = 0
\vdots
f_n(u_n,\ldots,u_3,u_2,u_1,p) = 0
```

Splitting the system in this form can have multiple advantages, including:

* Improved numerical stability and robustness of the solving process
* Improved performance due to using smaller Jacobians

The SCC-Split Nonlinear Problem is the ordered collection of nonlinear systems 
to solve in order solve the system in the optimized split form.

## Representation

The representation of the SCCNonlinearProblem is via an ordered collection
of `NonlinearProblem`s, `probs`, with an attached explicit function for pre-processing
a cache. This can be interpreted as follows:

```math
p_1 = g_1(u,p)
f_1(u_1,p_1) = 0
p_2 = g_2(u,p)
f_2(u_2,u_1,p_2) = 0
p_3 = g_3(u,p)
f_3(u_3,u_2,u_1,p_3) = 0
\vdots
p_n = g_n(u,p)
f_n(u_n,\ldots,u_3,u_2,u_1,p_n) = 0
```

where ``g_i`` is `explicitfuns![i]`. In a computational sense, `explictfuns!`
is instead a mutating function `explictfuns![i](prob.probs[i],sols)` which
updates the values of prob.probs[i] using the previous solutions `sols[i-1]`
and below.

!!! warn
    For the purposes of differentiation, it's assumed that `explictfuns!` does
    not modify tunable parameters!

!!! warn
    While `explictfuns![i]` could in theory use `sols[i+1]` in its computation,
    these values will not be updated. It is thus the contract of the interface
    to not use those values except for as caches to be overridden.

!!! note
    prob.probs[i].p can be aliased with each other as a performance / memory
    optimization. If they are aliased before the construction of the `probs`
    the runtime guarantees the aliasing behavior is kept.

## Example

For the following nonlinear problem:

```julia
function f(du,u,p)
	du[1] = cos(u[2]) - u[1]
	du[2] = sin(u[1] + u[2]) + u[2]
	du[3] = 2u[4] + u[3] + 1.0
	du[4] = u[5]^2 + u[4]
	du[5] = u[3]^2 + u[5]
	du[6] = u[1] + u[2] + u[3] + u[4] + u[5]    + 2.0u[6] + 2.5u[7] + 1.5u[8]
	du[7] = u[1] + u[2] + u[3] + 2.0u[4] + u[5] + 4.0u[6] - 1.5u[7] + 1.5u[8]
	du[8] = u[1] + 2.0u[2] + 3.0u[3] + 5.0u[4] + 6.0u[5] + u[6] - u[7] - u[8]
end
prob = NonlinearProblem(f, zeros(8))
sol = solve(prob)
```

The split SCC form is:

```julia
cache = zeros(3)

function f1(du,u,cache)
	du[1] = cos(u[2]) - u[1]
	du[2] = sin(u[1] + u[2]) + u[2]
end
explicitfun1(cache,sols) = nothing
prob1 = NonlinearProblem(NonlinearFunction{true, SciMLBase.NoSpecialize}(f1), zeros(2), cache)
sol1 = solve(prob1, NewtonRaphson())

function f2(du,u,cache)
	du[1] = 2u[2] + u[1] + 1.0
	du[2] = u[3]^2 + u[2]
	du[3] = u[1]^2 + u[3]
end
explicitfun2(cache,sols) = nothing
prob2 = NonlinearProblem(NonlinearFunction{true, SciMLBase.NoSpecialize}(f2), zeros(3), cache)
sol2 = solve(prob2, NewtonRaphson())

function f3(du,u,cache)
	du[1] = cache[1] + 2.0u[1] + 2.5u[2] + 1.5u[3]
	du[2] = cache[2] + 4.0u[1] - 1.5u[2] + 1.5u[3]
	du[3] = cache[3] + + u[1] - u[2] - u[3]
end
prob3 = NonlinearProblem(NonlinearFunction{true, SciMLBase.NoSpecialize}(f3), zeros(3), cache)
function explicitfun3(cache,sols)
	cache[1] = sols[1][1] + sols[1][2] + sols[2][1] + sols[2][2] + sols[2][3]
	cache[2] = sols[1][1] + sols[1][2] + sols[2][1] + 2.0sols[2][2] + sols[2][3]
	cache[3] = sols[1][1] + 2.0sols[1][2] + 3.0sols[2][1] + 5.0sols[2][2] + 6.0sols[2][3]
end
explicitfun3(cache,[sol1,sol2])
sol3 = solve(prob3, NewtonRaphson())
manualscc = [sol1; sol2; sol3]

sccprob = SciMLBase.SCCNonlinearProblem([prob1,prob2,prob3], SciMLBase.Void{Any}.([explicitfun1,explicitfun2,explicitfun3]))
```

Note that this example aliases the parameters together for a memory-reduced representation.

## Problem Type

### Constructors

### Fields

* `probs`: the collection of problems to solve
* `explictfuns!`: the explicit functions for mutating the parameter set
"""
mutable struct SCCNonlinearProblem{uType, iip, P, E, F <: NonlinearFunction{iip}, Par} <:
               AbstractNonlinearProblem{uType, iip}
    probs::P
    explicitfuns!::E
    # NonlinearFunction with `f = Returns(nothing)`
    f::F
    p::Par
    parameters_alias::Bool

    function SCCNonlinearProblem{P, E, F, Par}(probs::P, funs::E, f::F, pobj::Par,
            alias::Bool) where {P, E, F <: NonlinearFunction, Par}
        u0 = mapreduce(
            state_values, vcat, probs; init = similar(state_values(first(probs)), 0))
        uType = typeof(u0)
        new{uType, false, P, E, F, Par}(probs, funs, f, pobj, alias)
    end
end

function SCCNonlinearProblem(probs, explicitfuns!, parameter_object = nothing,
        parameters_alias = false; kwargs...)
    f = NonlinearFunction{false}(Returns(nothing); kwargs...)
    return SCCNonlinearProblem{typeof(probs), typeof(explicitfuns!),
        typeof(f), typeof(parameter_object)}(
        probs, explicitfuns!, f, parameter_object, parameters_alias)
end

function Base.getproperty(prob::SCCNonlinearProblem, name::Symbol)
    if name == :explictfuns!
        return getfield(prob, :explicitfuns!)
    elseif name == :ps
        return ParameterIndexingProxy(prob)
    end
    return getfield(prob, name)
end

function SymbolicIndexingInterface.symbolic_container(prob::SCCNonlinearProblem)
    prob.f
end
function SymbolicIndexingInterface.parameter_values(prob::SCCNonlinearProblem)
    prob.p
end
function SymbolicIndexingInterface.state_values(prob::SCCNonlinearProblem)
    mapreduce(
        state_values, vcat, prob.probs; init = similar(state_values(first(prob.probs)), 0))
end

function SymbolicIndexingInterface.set_state!(prob::SCCNonlinearProblem, val, idx)
    for scc in prob.probs
        svals = state_values(scc)
        checkbounds(Bool, svals, idx) && return set_state!(scc, val, idx)
        idx -= length(svals)
    end
    throw(BoundsError(state_values(prob), idx))
end

function SymbolicIndexingInterface.set_parameter!(prob::SCCNonlinearProblem, val, idx)
    if prob.p !== nothing
        set_parameter!(prob.p, val, idx)
        prob.parameters_alias && return
    end
    for scc in prob.probs
        is_parameter(scc, idx) || continue
        set_parameter!(scc, val, idx)
    end
end
