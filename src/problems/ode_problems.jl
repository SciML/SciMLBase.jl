"""
$(TYPEDEF)
"""
struct StandardODEProblem end

@doc doc"""

Defines an ordinary differential equation (ODE) problem.
Documentation Page: https://docs.sciml.ai/DiffEqDocs/stable/types/ode_types/

## Mathematical Specification of an ODE Problem

To define an ODE Problem, you simply need to give the function ``f`` and the initial
condition ``u_0`` which define an ODE:

```math
M \frac{du}{dt} = f(u,p,t)
```

There are two different ways of specifying `f`:
- `f(du,u,p,t)`: in-place. Memory-efficient when avoiding allocations. Best option for most cases unless mutation is not allowed.
- `f(u,p,t)`: returning `du`. Less memory-efficient way, particularly suitable when mutation is not allowed (e.g. with certain automatic differentiation packages such as Zygote).

`u₀` should be an AbstractArray (or number) whose geometry matches the desired geometry of `u`.
Note that we are not limited to numbers or vectors for `u₀`; one is allowed to
provide `u₀` as arbitrary matrices / higher dimension tensors as well.

For the mass matrix ``M``, see the documentation of `ODEFunction`.

## Problem Type

### Constructors

`ODEProblem` can be constructed by first building an `ODEFunction` or
by simply passing the ODE right-hand side to the constructor. The constructors
are:

- `ODEProblem(f::ODEFunction,u0,tspan,p=NullParameters();kwargs...)`
- `ODEProblem{isinplace,specialize}(f,u0,tspan,p=NullParameters();kwargs...)` :
  Defines the ODE with the specified functions. `isinplace` optionally sets whether
  the function is inplace or not. This is determined automatically, but not inferred.
  `specialize` optionally controls the specialization level. See the
  [specialization levels section of the SciMLBase documentation](https://docs.sciml.ai/SciMLBase/stable/interfaces/Problems/#Specialization-Levels)
  for more details. The default is `AutoSpecialize`.

For more details on the in-place and specialization controls, see the ODEFunction
documentation.

Parameters are optional, and if not given, then a `NullParameters()` singleton
will be used which will throw nice errors if you try to index non-existent
parameters. Any extra keyword arguments are passed on to the solvers. For example,
if you set a `callback` in the problem, then that `callback` will be added in
every solve call.

For specifying Jacobians and mass matrices, see the `ODEFunction` documentation.

### Fields

- `f`: The function in the ODE.
- `u0`: The initial condition.
- `tspan`: The timespan for the problem.
- `p`: The parameters.
- `kwargs`: The keyword arguments passed onto the solves.

## Example Problem

```julia
using SciMLBase
function lorenz!(du,u,p,t)
 du[1] = 10.0(u[2]-u[1])
 du[2] = u[1]*(28.0-u[3]) - u[2]
 du[3] = u[1]*u[2] - (8/3)*u[3]
end
u0 = [1.0;0.0;0.0]
tspan = (0.0,100.0)
prob = ODEProblem(lorenz!,u0,tspan)

# Test that it worked
using OrdinaryDiffEq
sol = solve(prob,Tsit5())
using Plots; plot(sol,vars=(1,2,3))
```

## More Example Problems

Example problems can be found in [DiffEqProblemLibrary.jl](https://github.com/SciML/DiffEqProblemLibrary.jl).

To use a sample problem, such as `prob_ode_linear`, you can do something like:

```julia
#] add ODEProblemLibrary
using ODEProblemLibrary
prob = ODEProblemLibrary.prob_ode_linear
sol = solve(prob)
```
"""
mutable struct ODEProblem{uType, tType, isinplace, P, F, K, PT} <:
       AbstractODEProblem{uType, tType, isinplace}
    """The ODE is `du = f(u,p,t)` for out-of-place and f(du,u,p,t) for in-place."""
    f::F
    """The initial condition is `u(tspan[1]) = u0`."""
    u0::uType
    """The solution `u(t)` will be computed for `tspan[1] ≤ t ≤ tspan[2]`."""
    tspan::tType
    """Constant parameters to be supplied as the second argument of `f`."""
    p::P
    """A callback to be applied to every solver which uses the problem."""
    kwargs::K
    """An internal argument for storing traits about the solving process."""
    problem_type::PT
    @add_kwonly function ODEProblem{iip}(f::AbstractODEFunction{iip},
        u0, tspan, p = NullParameters(),
        problem_type = StandardODEProblem();
        kwargs...) where {iip}
        _u0 = prepare_initial_state(u0)
        _tspan = promote_tspan(tspan)
        warn_paramtype(p)
        new{typeof(_u0), typeof(_tspan),
            isinplace(f), typeof(p), typeof(f),
            typeof(kwargs),
            typeof(problem_type)}(f,
            _u0,
            _tspan,
            p,
            kwargs,
            problem_type)
    end

    """
        ODEProblem{isinplace}(f,u0,tspan,p=NullParameters(),callback=CallbackSet())

    Define an ODE problem with the specified function.
    `isinplace` optionally sets whether the function is inplace or not.
    This is determined automatically, but not inferred.
    """
    function ODEProblem{iip}(f, u0, tspan, p = NullParameters(); kwargs...) where {iip}
        _u0 = prepare_initial_state(u0)
        _tspan = promote_tspan(tspan)
        _f = ODEFunction{iip, DEFAULT_SPECIALIZATION}(f)
        ODEProblem(_f, _u0, _tspan, p; kwargs...)
    end

    @add_kwonly function ODEProblem{iip, recompile}(f, u0, tspan, p = NullParameters();
        kwargs...) where {iip, recompile}
        ODEProblem{iip}(ODEFunction{iip, recompile}(f), u0, tspan, p; kwargs...)
    end

    function ODEProblem{iip, FunctionWrapperSpecialize}(f, u0, tspan, p = NullParameters();
        kwargs...) where {iip}
        _u0 = prepare_initial_state(u0)
        _tspan = promote_tspan(tspan)
        if !(f isa FunctionWrappersWrappers.FunctionWrappersWrapper)
            if iip
                ff = ODEFunction{iip, FunctionWrapperSpecialize}(wrapfun_iip(f,
                    (_u0, _u0, p,
                        _tspan[1])))
            else
                ff = ODEFunction{iip, FunctionWrapperSpecialize}(wrapfun_oop(f,
                    (_u0, p,
                        _tspan[1])))
            end
        end
        ODEProblem{iip}(ff, _u0, _tspan, p; kwargs...)
    end
end
TruncatedStacktraces.@truncate_stacktrace ODEProblem 3 1 2

function Base.setproperty!(prob::ODEProblem, s::Symbol, v)
    @warn "Mutation of ODEProblem detected. SciMLBase v2.0 has made ODEProblem temporarily mutable in order to allow for interfacing with EnzymeRules due to a current limitation in the rule system. This change is only intended to be temporary and ODEProblem will return to being a struct in a later non-breaking release. Do not rely on this behavior, use with caution."
    Base.setfield!(prob, s, v)
end

function Base.setproperty!(prob::ODEProblem, s::Symbol, v, order::Symbol)
    @warn "Mutation of ODEProblem detected. SciMLBase v2.0 has made ODEProblem temporarily mutable in order to allow for interfacing with EnzymeRules due to a current limitation in the rule system. This change is only intended to be temporary and ODEProblem will return to being a struct in a later non-breaking release. Do not rely on this behavior, use with caution."
    Base.setfield!(prob, s, v, order)
end

"""
    ODEProblem(f::ODEFunction,u0,tspan,p=NullParameters(),callback=CallbackSet())

Define an ODE problem from an [`ODEFunction`](@ref).
"""
function ODEProblem(f::AbstractODEFunction, u0, tspan, args...; kwargs...)
    ODEProblem{isinplace(f)}(f, u0, tspan, args...; kwargs...)
end

function ODEProblem(f, u0, tspan, p = NullParameters(); kwargs...)
    iip = isinplace(f, 4)
    _u0 = prepare_initial_state(u0)
    _tspan = promote_tspan(tspan)
    _f = ODEFunction{iip, DEFAULT_SPECIALIZATION}(f)
    ODEProblem(_f, _u0, _tspan, p; kwargs...)
end

"""
$(TYPEDEF)
"""
abstract type AbstractDynamicalODEProblem end

@doc doc"""

Defines a dynamical ordinary differential equation (ODE) problem.
Documentation Page: https://docs.sciml.ai/DiffEqDocs/stable/types/dynamical_types/

Dynamical ordinary differential equations, such as those arising from the definition
of a Hamiltonian system or a second order ODE, have a special structure that can be
utilized in the solution of the differential equation. On this page, we describe
how to define second order differential equations for their efficient numerical solution.

## Mathematical Specification of a Dynamical ODE Problem

These algorithms require a Partitioned ODE of the form:

```math
\frac{dv}{dt} = f_1(u,t) \\
\frac{du}{dt} = f_2(v) \\
```
This is a Partitioned ODE partitioned into two groups, so the functions should be
specified as `f1(dv,v,u,p,t)` and `f2(du,v,u,p,t)` (in the inplace form), where `f1`
is independent of `v` (unless specified by the solver), and `f2` is independent
of `u` and `t`. This includes discretizations arising from
`SecondOrderODEProblem`s where the velocity is not used in the acceleration function,
and Hamiltonians where the potential is (or can be) time-dependent, but the kinetic
energy is only dependent on `v`.

Note that some methods assume that the integral of `f2` is a quadratic form. That
means that `f2=v'*M*v`, i.e. ``\int f_2 = \frac{1}{2} m v^2``, giving `du = v`.
This is equivalent to saying that the kinetic energy is related to ``v^2``. The
methods which require this assumption will lose accuracy if this assumption is
violated. Methods listed make note of this requirement with "Requires
quadratic kinetic energy".

### Constructor

```julia
DynamicalODEProblem(f::DynamicalODEFunction,v0,u0,tspan,p=NullParameters();kwargs...)
DynamicalODEProblem{isinplace}(f1,f2,v0,u0,tspan,p=NullParameters();kwargs...)
```

Defines the ODE with the specified functions. `isinplace` optionally sets whether
the function is inplace or not. This is determined automatically, but not inferred.

Parameters are optional, and if not given, then a `NullParameters()` singleton
will be used which will throw nice errors if you try to index non-existent
parameters. Any extra keyword arguments are passed on to the solvers. For example,
if you set a `callback` in the problem, then that `callback` will be added in
every solve call.

### Fields

* `f1` and `f2`: The functions in the ODE.
* `v0` and `u0`: The initial conditions.
* `tspan`: The timespan for the problem.
* `p`: The parameters for the problem. Defaults to `NullParameters`
* `kwargs`: The keyword arguments passed onto the solves.
"""
struct DynamicalODEProblem{iip} <: AbstractDynamicalODEProblem end

"""
    DynamicalODEProblem(f::DynamicalODEFunction,v0,u0,tspan,p=NullParameters(),callback=CallbackSet())

Define a dynamical ODE function from a [`DynamicalODEFunction`](@ref).
"""
function DynamicalODEProblem(f::DynamicalODEFunction, du0, u0, tspan, p = NullParameters();
    kwargs...)
    ODEProblem(f, ArrayPartition(du0, u0), tspan, p; kwargs...)
end
function DynamicalODEProblem(f1, f2, du0, u0, tspan, p = NullParameters(); kwargs...)
    ODEProblem(DynamicalODEFunction(f1, f2), ArrayPartition(du0, u0), tspan, p; kwargs...)
end

function DynamicalODEProblem{iip}(f1, f2, du0, u0, tspan, p = NullParameters();
    kwargs...) where {iip}
    ODEProblem(DynamicalODEFunction{iip}(f1, f2), ArrayPartition(du0, u0), tspan, p,
        DynamicalODEProblem{iip}(); kwargs...)
end

@doc doc"""

Defines a second order ordinary differential equation (ODE) problem.
Documentation Page: https://docs.sciml.ai/DiffEqDocs/stable/types/dynamical_types/

## Mathematical Specification of a 2nd Order ODE Problem

To define a 2nd Order ODE Problem, you simply need to give the function ``f``
and the initial condition ``u_0`` which define an ODE:

```math
u'' = f(u',u,p,t)
```

`f` should be specified as `f(du,u,p,t)` (or in-place as `f(ddu,du,u,p,t)`), and `u₀`
should be an AbstractArray (or number) whose geometry matches the desired
geometry of `u`. Note that we are not limited to numbers or vectors for `u₀`;
one is allowed to provide `u₀` as arbitrary matrices / higher dimension tensors
as well.

From this form, a dynamical ODE:

```math
v' = f(v,u,p,t) \\
u' = v \\
```

is generated.

### Constructors

```julia
SecondOrderODEProblem{isinplace}(f,du0,u0,tspan,callback=CallbackSet())
```

Defines the ODE with the specified functions.

### Fields

* `f`: The function for the second derivative.
* `du0`: The initial derivative.
* `u0`: The initial condition.
* `tspan`: The timespan for the problem.
* `callback`: A callback to be applied to every solver which uses the problem.
  Defaults to nothing.
"""
struct SecondOrderODEProblem{iip} <: AbstractDynamicalODEProblem end
function SecondOrderODEProblem(f, du0, u0, tspan, p = NullParameters(); kwargs...)
    iip = isinplace(f, 5)
    SecondOrderODEProblem{iip}(f, du0, u0, tspan, p; kwargs...)
end

function SecondOrderODEProblem{iip}(f, du0, u0, tspan, p = NullParameters();
    kwargs...) where {iip}
    if iip
        f2 = function (du, v, u, p, t)
            du .= v
        end
    else
        f2 = function (v, u, p, t)
            v
        end
    end
    _u0 = ArrayPartition((du0, u0))
    ODEProblem(DynamicalODEFunction{iip}(f, f2), _u0, tspan, p,
        SecondOrderODEProblem{iip}(); kwargs...)
end
function SecondOrderODEProblem(f::DynamicalODEFunction, du0, u0, tspan,
    p = NullParameters(); kwargs...)
    iip = isinplace(f.f1, 5)
    _u0 = ArrayPartition((du0, u0))
    if f.f2.f === nothing
        if iip
            f2 = function (du, v, u, p, t)
                du .= v
            end
        else
            f2 = function (v, u, p, t)
                v
            end
        end
        return ODEProblem(DynamicalODEFunction{iip}(f.f1, f2; mass_matrix = f.mass_matrix,
                analytic = f.analytic), _u0, tspan, p,
            SecondOrderODEProblem{iip}(); kwargs...)
    else
        return ODEProblem(DynamicalODEFunction{iip}(f.f1, f.f2; mass_matrix = f.mass_matrix,
                analytic = f.analytic), _u0, tspan, p,
            SecondOrderODEProblem{iip}(); kwargs...)
    end
end

"""
$(TYPEDEF)
"""
abstract type AbstractSplitODEProblem end

@doc doc"""

Defines a split ordinary differential equation (ODE) problem.
Documentation Page: https://docs.sciml.ai/DiffEqDocs/stable/types/split_ode_types/

## Mathematical Specification of a Split ODE Problem

To define a `SplitODEProblem`, you simply need to give two functions
``f_1`` and ``f_2`` along with an initial condition ``u_0`` which
define an ODE:

```math
\frac{du}{dt} =  f_1(u,p,t) + f_2(u,p,t)
```

`f` should be specified as `f(u,p,t)` (or in-place as `f(du,u,p,t)`), and `u₀` should
be an AbstractArray (or number) whose geometry matches the desired geometry of `u`.
Note that we are not limited to numbers or vectors for `u₀`; one is allowed to
provide `u₀` as arbitrary matrices / higher dimension tensors as well.

Many splits are at least partially linear. That is the equation:

```math
\frac{du}{dt} =  Au + f_2(u,p,t)
```

For how to define a linear function `A`, see the documentation for the [AbstractSciMLOperator](https://docs.sciml.ai/SciMLOperators/stable/interface/).

### Constructors

```julia
SplitODEProblem(f::SplitFunction,u0,tspan,p=NullParameters();kwargs...)
SplitODEProblem{isinplace}(f1,f2,u0,tspan,p=NullParameters();kwargs...)
```

The `isinplace` parameter can be omitted and will be determined using the signature of `f2`.
Note that both `f1` and `f2` should support the in-place style if `isinplace` is `true` or they
should both support the out-of-place style if `isinplace` is `false`. You cannot mix up the two styles.

Parameters are optional, and if not given, then a `NullParameters()` singleton
will be used which will throw nice errors if you try to index non-existent
parameters. Any extra keyword arguments are passed on to the solvers. For example,
if you set a `callback` in the problem, then that `callback` will be added in
every solve call.

Under the hood, a `SplitODEProblem` is just a regular `ODEProblem` whose `f` is a `SplitFunction`.
Therefore, you can solve a `SplitODEProblem` using the same solvers for `ODEProblem`. For solvers
dedicated to split problems, see [Split ODE Solvers](@ref split_ode_solve).

For specifying Jacobians and mass matrices, see the
[DiffEqFunctions](@ref performance_overloads)
page.

### Fields

* `f1`, `f2`: The functions in the ODE.
* `u0`: The initial condition.
* `tspan`: The timespan for the problem.
* `p`: The parameters for the problem. Defaults to `NullParameters`
* `kwargs`: The keyword arguments passed onto the solves.
"""
struct SplitODEProblem{iip} <: AbstractSplitODEProblem end

function SplitODEProblem(f1, f2, u0, tspan, p = NullParameters(); kwargs...)
    f = SplitFunction(f1, f2)
    SplitODEProblem(f, u0, tspan, p; kwargs...)
end

function SplitODEProblem{iip}(f1, f2, u0, tspan, p = NullParameters();
    kwargs...) where {iip}
    f = SplitFunction{iip}(f1, f2)
    SplitODEProblem(f, u0, tspan, p; kwargs...)
end

"""
$(SIGNATURES)

Define a split ODE problem from a [`SplitFunction`](@ref).
"""
function SplitODEProblem(f::SplitFunction, u0, tspan, p = NullParameters(); kwargs...)
    SplitODEProblem{isinplace(f)}(f, u0, tspan, p; kwargs...)
end
function SplitODEProblem{iip}(f::SplitFunction, u0, tspan, p = NullParameters();
    kwargs...) where {iip}
    if f.cache === nothing && iip
        cache = similar(u0)
        f = SplitFunction{iip}(f.f1, f.f2; mass_matrix = f.mass_matrix,
            _func_cache = cache, analytic = f.analytic)
    end
    ODEProblem(f, u0, tspan, p, SplitODEProblem{iip}(); kwargs...)
end

abstract type AbstractIncrementingODEProblem end

"""
$(SIGNATURES)

Experimental
"""
struct IncrementingODEProblem{iip} <: AbstractIncrementingODEProblem end

function IncrementingODEProblem(f, u0, tspan, p = NullParameters(); kwargs...)
    f = IncrementingODEFunction(f)
    IncrementingODEProblem(f, u0, tspan, p; kwargs...)
end

function IncrementingODEProblem{iip}(f, u0, tspan, p = NullParameters();
    kwargs...) where {iip}
    f = IncrementingODEFunction{iip}(f)
    IncrementingODEProblem(f, u0, tspan, p; kwargs...)
end

function IncrementingODEProblem(f::IncrementingODEFunction, u0, tspan, p = NullParameters();
    kwargs...)
    IncrementingODEProblem{isinplace(f)}(f, u0, tspan, p; kwargs...)
end

function IncrementingODEProblem{iip}(f::IncrementingODEFunction, u0, tspan,
    p = NullParameters(); kwargs...) where {iip}
    ODEProblem(f, u0, tspan, p, IncrementingODEProblem{iip}(); kwargs...)
end
