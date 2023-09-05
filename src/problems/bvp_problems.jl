"""
$(TYPEDEF)
"""
struct StandardBVProblem end

@doc doc"""

Defines an BVP problem.
Documentation Page: https://docs.sciml.ai/DiffEqDocs/stable/types/bvp_types/

## Mathematical Specification of a BVP Problem

To define a BVP Problem, you simply need to give the function ``f`` and the initial
condition ``u_0`` which define an ODE:

```math
\frac{du}{dt} = f(u,p,t)
```

along with an implicit function `bc!` which defines the residual equation, where

```math
bc(u,p,t) = 0
```

is the manifold on which the solution must live. A common form for this is the
two-point `BVProblem` where the manifold defines the solution at two points:

```math
u(t_0) = a
u(t_f) = b
```

## Problem Type

### Constructors

```julia
TwoPointBVProblem{isinplace}(f,bc!,u0,tspan,p=NullParameters();kwargs...)
BVProblem{isinplace}(f,bc!,u0,tspan,p=NullParameters();kwargs...)
```

or if we have an initial guess function `initialGuess(t)` for the given BVP,
we can pass the initial guess to the problem constructors:

```julia
TwoPointBVProblem{isinplace}(f,bc!,initialGuess,tspan,p=NullParameters();kwargs...)
BVProblem{isinplace}(f,bc!,initialGuess,tspan,p=NullParameters();kwargs...)
```

For any BVP problem type, `bc!` is the inplace function:

```julia
bc!(residual, u, p, t)
```

where `residual` computed from the current `u`. `u` is an array of solution values
where `u[i]` is at time `t[i]`, while `p` are the parameters. For a `TwoPointBVProblem`,
`t = tspan`. For the more general `BVProblem`, `u` can be all of the internal
time points, and for shooting type methods `u=sol` the ODE solution.
Note that all features of the `ODESolution` are present in this form.
In both cases, the size of the residual matches the size of the initial condition.

Parameters are optional, and if not given, then a `NullParameters()` singleton
will be used which will throw nice errors if you try to index non-existent
parameters. Any extra keyword arguments are passed on to the solvers. For example,
if you set a `callback` in the problem, then that `callback` will be added in
every solve call.

### Fields

* `f`: The function for the ODE.
* `bc`: The boundary condition function.
* `u0`: The initial condition. Either the initial condition for the ODE as an
  initial value problem, or a `Vector` of values for ``u(t_i)`` for collocation
  methods.
* `tspan`: The timespan for the problem.
* `p`: The parameters for the problem. Defaults to `NullParameters`
* `kwargs`: The keyword arguments passed onto the solves.
"""
struct BVProblem{uType, tType, isinplace, P, F, BF, PT, K} <:
       AbstractBVProblem{uType, tType, isinplace}
    f::F
    bc::BF
    u0::uType
    tspan::tType
    p::P
    problem_type::PT
    kwargs::K

    @add_kwonly function BVProblem{iip}(f::AbstractBVPFunction{iip}, bc, u0, tspan,
        p = NullParameters(),
        problem_type = StandardBVProblem();
        kwargs...) where {iip}
        _tspan = promote_tspan(tspan)
        warn_paramtype(p)
        new{typeof(u0), typeof(_tspan), iip, typeof(p),
            typeof(f.f), typeof(bc),
            typeof(problem_type), typeof(kwargs)}(f.f, bc, u0, _tspan, p,
            problem_type, kwargs)
    end

    function BVProblem{iip}(f, bc, u0, tspan, p = NullParameters(); kwargs...) where {iip}
        BVProblem(BVPFunction{iip}(f, bc), bc, u0, tspan, p; kwargs...)
    end
end

TruncatedStacktraces.@truncate_stacktrace BVProblem 3 1 2

function BVProblem(f, bc, u0, tspan, p = NullParameters(); kwargs...)
    BVProblem(BVPFunction(f, bc), u0, tspan, p; kwargs...)
end

function BVProblem(f::AbstractBVPFunction, u0, tspan, p = NullParameters(); kwargs...)
    BVProblem{isinplace(f)}(f.f, f.bc, u0, tspan, p; kwargs...)
end

"""
$(TYPEDEF)
"""
struct TwoPointBVPFunction{bF}
    bc::bF
end
TwoPointBVPFunction(; bc = error("No argument `bc`")) = TwoPointBVPFunction(bc)
function (f::TwoPointBVPFunction)(residuala, residualb, ua, ub, p)
    return f.bc(residuala, residualb, ua, ub, p)
end
function (f::TwoPointBVPFunction)(residual::Tuple, u, p)
    return f(residual[1], residual[2], u[1], u[end], p)
end

"""
$(TYPEDEF)
"""
struct TwoPointBVProblem{iip} end
function TwoPointBVProblem(f, bc, u0, tspan, p = NullParameters(); kwargs...)
    iip = isinplace(f, 4)
    TwoPointBVProblem{iip}(f, bc, u0, tspan, p; kwargs...)
end
function TwoPointBVProblem{iip}(f, bc, u0, tspan, p = NullParameters();
    kwargs...) where {iip}
    BVProblem{iip}(f, TwoPointBVPFunction(bc), u0, tspan, p; kwargs...)
end

# Allow previous timeseries solution
function TwoPointBVProblem(f::AbstractODEFunction,
    bc,
    sol::T,
    tspan::Tuple,
    p = NullParameters()) where {T <: AbstractTimeseriesSolution}
    TwoPointBVProblem(f, bc, sol.u, tspan, p)
end
# Allow initial guess function for the initial guess
function TwoPointBVProblem(f::AbstractODEFunction,
    bc,
    initialGuess,
    tspan::AbstractVector,
    p = NullParameters();
    kwargs...)
    u0 = [initialGuess(i) for i in tspan]
    TwoPointBVProblem(f, bc, u0, (tspan[1], tspan[end]), p)
end
