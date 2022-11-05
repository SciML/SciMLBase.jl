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

Parameters are optional, and if not given then a `NullParameters()` singleton
will be used which will throw nice errors if you try to index non-existent
parameters. Any extra keyword arguments are passed on to the solvers. For example,
if you set a `callback` in the problem, then that `callback` will be added in
every solve call.

### Fields

* `f`: The function for the ODE.
* `bc`: The boundary condition function.
* `u0`: The initial condition. Either the initial condition for the ODE as an
  initial value problem, or a `Vector` of values for ``u(t_i)`` for collocation
  methods
* `tspan`: The timespan for the problem.
* `p`: The parameters for the problem. Defaults to `NullParameters`
* `kwargs`: The keyword arguments passed onto the solves.
"""
struct BVProblem{uType, tType, isinplace, P, F, bF, PT, K} <:
       AbstractBVProblem{uType, tType, isinplace}
    f::F
    bc::bF
    u0::uType
    tspan::tType
    p::P
    problem_type::PT
    kwargs::K
    @add_kwonly function BVProblem{iip}(f::AbstractODEFunction, bc, u0, tspan,
                                        p = NullParameters(),
                                        problem_type = StandardBVProblem();
                                        kwargs...) where {iip}
        _tspan = promote_tspan(tspan)
        new{typeof(u0), typeof(tspan), iip, typeof(p),
            typeof(f), typeof(bc),
            typeof(problem_type), typeof(kwargs)}(f, bc, u0, _tspan, p,
                                                  problem_type, kwargs)
    end

    function BVProblem{iip}(f, bc, u0, tspan, p = NullParameters(); kwargs...) where {iip}
        BVProblem(ODEFunction{iip}(f), bc, u0, tspan, p; kwargs...)
    end
end

function BVProblem(f::AbstractODEFunction, bc, u0, tspan, args...; kwargs...)
    BVProblem{isinplace(f, 4)}(f, bc, u0, tspan, args...; kwargs...)
end

function BVProblem(f, bc, u0, tspan, p = NullParameters(); kwargs...)
    BVProblem(ODEFunction(f), bc, u0, tspan, p; kwargs...)
end

# convenience interfaces:
# Allow any previous timeseries solution
function BVProblem(f::AbstractODEFunction, bc, sol::T, tspan::Tuple, p = NullParameters();
                   kwargs...) where {T <: AbstractTimeseriesSolution}
    BVProblem(f, bc, sol.u, tspan, p)
end
# Allow a function of time for the initial guess
function BVProblem(f::AbstractODEFunction, bc, initialGuess, tspan::AbstractVector,
                   p = NullParameters(); kwargs...)
    u0 = [initialGuess(i) for i in tspan]
    BVProblem(f, bc, u0, (tspan[1], tspan[end]), p)
end

"""
$(TYPEDEF)
"""
struct TwoPointBVPFunction{bF}
    bc::bF
end
TwoPointBVPFunction(; bc = error("No argument bc")) = TwoPointBVPFunction(bc)
(f::TwoPointBVPFunction)(residual, ua, ub, p) = f.bc(residual, ua, ub, p)
(f::TwoPointBVPFunction)(residual, u, p) = f.bc(residual, u[1], u[end], p)

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
