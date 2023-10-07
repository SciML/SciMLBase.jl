"""
$(TYPEDEF)
"""
struct StandardDDEProblem end

@doc doc"""

Defines a delay differential equation (DDE) problem.
Documentation Page: https://docs.sciml.ai/DiffEqDocs/stable/types/dde_types/

## Mathematical Specification of a DDE Problem

To define a DDE Problem, you simply need to give the function ``f``, the initial
condition ``u_0`` at time point ``t_0``, and the history function ``h``
which together define a DDE:

```math
\frac{du}{dt} = f(u,h,p,t) \qquad (t \geq t_0)
```
```math
u(t_0) = u_0,
```
```math
u(t) = h(t) \qquad (t < t_0).
```

``f`` should be specified as `f(u, h, p, t)` (or in-place as `f(du, u, h, p, t)`),
``u_0`` should be an AbstractArray (or number) whose geometry matches the
desired geometry of `u`, and ``h`` should be specified as described below. The
history function `h` is accessed for all delayed values. Note that we are not
limited to numbers or vectors for ``u_0``; one is allowed to provide ``u_0``
as arbitrary matrices / higher dimension tensors as well.

## Functional Forms of the History Function

The history function `h` can be called in the following ways:

- `h(p, t)`: out-of-place calculation
- `h(out, p, t)`: in-place calculation
- `h(p, t, deriv::Type{Val{i}})`: out-of-place calculation of the `i`th derivative
- `h(out, p, t, deriv::Type{Val{i}})`: in-place calculation of the `i`th derivative
- `h(args...; idxs)`: calculation of `h(args...)` for indices `idxs`

Note that a dispatch for the supplied history function of matching form is required
for whichever function forms are used in the user derivative function `f`.

## Declaring Lags

Lags are declared separately from their use. One can use any lag by simply using
the interpolant of `h` at that point. However, one should use caution in order
to achieve the best accuracy. When lags are declared, the solvers can be more
efficient and accurate, and this is thus recommended.

## Neutral and Retarded Delay Differential Equations

Note that the history function specification can be used to specify general
retarded arguments, i.e. `h(p,α(u,t))`. Neutral delay differential equations
can be specified by using the `deriv` value in the history interpolation.
For example, `h(p,t-τ, Val{1})` returns the first derivative of the history
values at time `t-τ`.

Note that algebraic equations can be specified by using a singular mass matrix.

## Problem Type

### Constructors

```
DDEProblem(f[, u0], h, tspan[, p]; <keyword arguments>)
DDEProblem{isinplace,specialize}(f[, u0], h, tspan[, p]; <keyword arguments>)
```

`isinplace` optionally sets whether the function is inplace or not. This is
determined automatically, but not inferred. `specialize` optionally controls
the specialization level. See the [specialization levels section of the SciMLBase documentation](https://docs.sciml.ai/SciMLBase/stable/interfaces/Problems/#Specialization-Levels)
for more details. The default is `AutoSpecialize`.

For more details on the in-place and specialization controls, see the ODEFunction
documentation.

Parameters are optional, and if not given, then a `NullParameters()` singleton
will be used which will throw nice errors if you try to index non-existent
parameters. Any extra keyword arguments are passed on to the solvers. For example,
if you set a `callback` in the problem, then that `callback` will be added in
every solve call.

For specifying Jacobians and mass matrices, see the [DiffEqFunctions](@ref performance_overloads) page.

### Arguments

* `f`: The function in the DDE.
* `u0`: The initial condition. Defaults to the value `h(p, first(tspan))` of the history function evaluated at the initial time point.
* `h`: The history function for the DDE before `t0`.
* `tspan`: The timespan for the problem.
* `p`: The parameters with which function `f` is called. Defaults to `NullParameters`.
* `constant_lags`: A collection of constant lags used by the history function `h`. Defaults to `()`.
* `dependent_lags` A tuple of functions `(u, p, t) -> lag` for the state-dependent lags
  used by the history function `h`. Defaults to `()`.
* `neutral`: If the DDE is neutral, i.e., if delays appear in derivative terms.
* `order_discontinuity_t0`: The order of the discontinuity at the initial time point. Defaults to `0` if an initial condition `u0` is provided. Otherwise, it is forced to be greater or equal than `1`.
* `kwargs`: The keyword arguments passed onto the solves.

## Dynamical Delay Differential Equations

Much like [Dynamical ODEs](@ref dynamical_prob), a Dynamical DDE is a Partitioned DDE
of the form:

```math
\frac{dv}{dt} = f_1(u,t,h) \\
\frac{du}{dt} = f_2(v,h) \\
```

### Constructors

```
DynamicalDDEProblem(f1, f2[, v0, u0], h, tspan[, p]; <keyword arguments>)
DynamicalDDEProblem{isinplace}(f1, f2[, v0, u0], h, tspan[, p]; <keyword arguments>)
```
Parameter `isinplace` optionally sets whether the function is inplace or not.
This is determined automatically, but not inferred.

### Arguments

* `f`: The function in the DDE.
* `v0` and `u0`: The initial condition. Defaults to the values `h(p, first(tspan))...` of the history function evaluated at the initial time point.
* `h`: The history function for the DDE before `t0`. Must return an object with the indices 1 and 2, with the values of `v` and `u` respectively.
* `tspan`: The timespan for the problem.
* `p`: The parameters with which function `f` is called. Defaults to `NullParameters`.
* `constant_lags`: A collection of constant lags used by the history function `h`. Defaults to `()`.
* `dependent_lags` A tuple of functions `(v, u, p, t) -> lag` for the state-dependent lags
  used by the history function `h`. Defaults to `()`.
* `neutral`: If the DDE is neutral, i.e., if delays appear in derivative terms.
* `order_discontinuity_t0`: The order of the discontinuity at the initial time point. Defaults to `0` if an initial condition `u0` is provided. Otherwise, it is forced to be greater or equal than `1`.
* `kwargs`: The keyword arguments passed onto the solves.

For dynamical and second order DDEs, the history function will return an object with
the indices 1 and 2 defined, where `h(p, t_prev)[1]` is the value of ``f_2(v, u, h, p,
t_{\mathrm{prev}})`` and `h(p, t_prev)[2]` is the value of ``f_1(v, u, h, p, t_{\mathrm{prev}})``
(this is for consistency with the ordering of the initial conditions in the constructor).
The supplied history function must also return such a 2-index object, which can be accomplished
with a tuple `(v,u)` or vector `[v,u]`.

## 2nd Order Delay Differential Equations

To define a 2nd Order DDE Problem, you simply need to give the function ``f``
and the initial condition ``u_0`` which define an DDE:

```math
u'' = f(u',u,h,p,t)
```

`f` should be specified as `f(du,u,p,t)` (or in-place as `f(ddu,du,u,p,t)`), and `u₀`
should be an AbstractArray (or number) whose geometry matches the desired
geometry of `u`. Note that we are not limited to numbers or vectors for `u₀`;
one is allowed to provide `u₀` as arbitrary matrices / higher dimension tensors
as well.

From this form, a dynamical ODE:

```math
v' = f(v,u,h,p,t) \\
u' = v \\
```

### Constructors

```
SecondOrderDDEProblem(f, [, du0, u0], h, tspan[, p]; <keyword arguments>)
SecondOrderDDEProblem{isinplace}(f, [, du0, u0], h, tspan[, p]; <keyword arguments>)
```

Parameter `isinplace` optionally sets whether the function is inplace or not.
This is determined automatically, but not inferred.

### Arguments

* `f`: The function in the DDE.
* `du0` and `u0`: The initial condition. Defaults to the values `h(p, first(tspan))...` of the history function evaluated at the initial time point.
* `h`: The history function for the DDE before `t0`. Must return an object with the indices 1 and 2, with the values of `v` and `u` respectively.
* `tspan`: The timespan for the problem.
* `p`: The parameters with which function `f` is called. Defaults to `NullParameters`.
* `constant_lags`: A collection of constant lags used by the history function `h`. Defaults to `()`.
* `dependent_lags` A tuple of functions `(v, u, p, t) -> lag` for the state-dependent lags
  used by the history function `h`. Defaults to `()`.
* `neutral`: If the DDE is neutral, i.e., if delays appear in derivative terms.
* `order_discontinuity_t0`: The order of the discontinuity at the initial time point. Defaults to `0` if an initial condition `u0` is provided. Otherwise, it is forced to be greater or equal than `1`.
* `kwargs`: The keyword arguments passed onto the solves.

As above, the history function will return an object with indices 1 and 2, with the values of `du` and `u` respectively. The supplied history function must also match this return type, e.g. by returning a 2-element tuple or vector.

## Example Problems

Example problems can be found in [DiffEqProblemLibrary.jl](https://github.com/SciML/DiffEqProblemLibrary.jl).

To use a sample problem, such as `prob_dde_constant_1delay_ip`, you can do something like:

```julia
#] add DDEProblemLibrary
using DDEProblemLibrary
prob = DDEProblemLibrary.prob_dde_constant_1delay_ip
sol = solve(prob)
```
"""
struct DDEProblem{uType, tType, lType, lType2, isinplace, P, F, H, K, PT} <:
       AbstractDDEProblem{uType, tType, lType, isinplace}
    f::F
    u0::uType
    h::H
    tspan::tType
    p::P
    constant_lags::lType
    dependent_lags::lType2
    kwargs::K
    neutral::Bool
    order_discontinuity_t0::Int
    problem_type::PT

    @add_kwonly function DDEProblem{iip}(f::AbstractDDEFunction{iip}, u0, h, tspan,
        p = NullParameters();
        constant_lags = (),
        dependent_lags = (),
        neutral = f.mass_matrix !== I &&
                  det(f.mass_matrix) != 1,
        order_discontinuity_t0 = 0,
        problem_type = StandardDDEProblem(),
        kwargs...) where {iip}
        _u0 = prepare_initial_state(u0)
        _tspan = promote_tspan(tspan)
        warn_paramtype(p)
        new{typeof(_u0), typeof(_tspan), typeof(constant_lags), typeof(dependent_lags),
            isinplace(f),
            typeof(p), typeof(f), typeof(h), typeof(kwargs), typeof(problem_type)}(f, _u0,
            h,
            _tspan,
            p,
            constant_lags,
            dependent_lags,
            kwargs,
            neutral,
            order_discontinuity_t0,
            problem_type)
    end

    function DDEProblem{iip}(f::AbstractDDEFunction{iip}, h, tspan::Tuple,
        p = NullParameters();
        order_discontinuity_t0 = 1, kwargs...) where {iip}
        DDEProblem{iip}(f, h(p, first(tspan)), h, tspan, p;
            order_discontinuity_t0 = max(1, order_discontinuity_t0), kwargs...)
    end

    function DDEProblem{iip}(f, args...; kwargs...) where {iip}
        DDEProblem{iip}(DDEFunction{iip}(f), args...; kwargs...)
    end
end

TruncatedStacktraces.@truncate_stacktrace DDEProblem 5 1 2

DDEProblem(f, args...; kwargs...) = DDEProblem(DDEFunction(f), args...; kwargs...)

function DDEProblem(f::AbstractDDEFunction, args...; kwargs...)
    DDEProblem{isinplace(f)}(f, args...; kwargs...)
end

"""
$(TYPEDEF)
"""
abstract type AbstractDynamicalDDEProblem end

"""
$(TYPEDEF)
"""
struct DynamicalDDEProblem{iip} <: AbstractDynamicalDDEProblem end

# u' = f1(v,h)
# v' = f2(t,u,h)
"""
    DynamicalDDEProblem(f::DynamicalDDEFunction,v0,u0,tspan,p=NullParameters(),callback=CallbackSet())

Define a dynamical DDE problem from a [`DynamicalDDEFunction`](@ref).
"""
function DynamicalDDEProblem(f::DynamicalDDEFunction, v0, u0, h, tspan,
    p = NullParameters(); dependent_lags = (), kwargs...)
    DDEProblem(f, ArrayPartition(v0, u0), h, tspan, p;
        problem_type = DynamicalDDEProblem{isinplace(f)}(),
        dependent_lags = ntuple(i -> (u, p, t) -> dependent_lags[i](u[1], u[2], p, t),
            length(dependent_lags)),
        kwargs...)
end
function DynamicalDDEProblem(f::DynamicalDDEFunction, h, tspan, p = NullParameters();
    kwargs...)
    DynamicalDDEProblem(f, h(p, first(tspan))..., h, tspan, p; kwargs...)
end
function DynamicalDDEProblem(f1, f2, args...; kwargs...)
    DynamicalDDEProblem(DynamicalDDEFunction(f1, f2), args...; kwargs...)
end

"""
    DynamicalDDEProblem{isinplace}(f1,f2,v0,u0,h,tspan,p=NullParameters(),callback=CallbackSet())

Define a dynamical DDE problem from the two functions `f1` and `f2`.

# Arguments
* `f1` and `f2`: The functions in the DDE.
* `v0` and `u0`: The initial conditions.
* `h`: The initial history function.
* `tspan`: The timespan for the problem.
* `p`: Parameter values for `f1` and `f2`.
* `callback`: A callback to be applied to every solver which uses the problem. Defaults to nothing.

`isinplace` optionally sets whether the function is inplace or not.
This is determined automatically, but not inferred.
"""
function DynamicalDDEProblem{iip}(f1, f2, args...; kwargs...) where {iip}
    DynamicalDDEProblem(DynamicalDDEFunction{iip}(f1, f2), args...; kwargs...)
end

# u'' = f(du,u,h,p,t)
"""
$(TYPEDEF)
"""
struct SecondOrderDDEProblem{iip} <: AbstractDynamicalDDEProblem end
function SecondOrderDDEProblem(f, args...; kwargs...)
    iip = isinplace(f, 6)
    SecondOrderDDEProblem{iip}(f, args...; kwargs...)
end

"""
    SecondOrderDDEProblem{isinplace}(f,du0,u0,h,tspan,p=NullParameters(),callback=CallbackSet())

Define a second order DDE problem with the specified function.

# Arguments
* `f`: The function for the second derivative.
* `du0`: The initial derivative.
* `u0`: The initial condition.
* `h`: The initial history function.
* `tspan`: The timespan for the problem.
* `p`: Parameter values for `f`.
* `callback`: A callback to be applied to every solver which uses the problem. Defaults to nothing.

`isinplace` optionally sets whether the function is inplace or not.
This is determined automatically, but not inferred.
"""
function SecondOrderDDEProblem{iip}(f, args...; kwargs...) where {iip}
    if iip
        f2 = function (du, v, u, h, p, t)
            du .= v
        end
    else
        f2 = function (v, u, h, p, t)
            v
        end
    end
    DynamicalDDEProblem{iip}(f, f2, args...; problem_type = SecondOrderDDEProblem{iip}(),
        kwargs...)
end
function SecondOrderDDEProblem(f::DynamicalDDEFunction, args...; kwargs...)
    iip = isinplace(f.f1, 6)
    if f.f2.f === nothing
        if iip
            f2 = function (du, v, u, h, p, t)
                du .= v
            end
        else
            f2 = function (v, u, h, p, t)
                v
            end
        end
        return DynamicalDDEProblem(DynamicalDDEFunction{iip}(f.f1, f2;
                mass_matrix = f.mass_matrix,
                analytic = f.analytic),
            args...; problem_type = SecondOrderDDEProblem{iip}(),
            kwargs...)
    else
        return DynamicalDDEProblem(DynamicalDDEFunction{iip}(f.f1, f.f2;
                mass_matrix = f.mass_matrix,
                analytic = f.analytic),
            args...; problem_type = SecondOrderDDEProblem{iip}(),
            kwargs...)
    end
end
