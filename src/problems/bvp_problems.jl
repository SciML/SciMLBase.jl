"""
$(TYPEDEF)
"""
struct StandardBVProblem end

"""
$(TYPEDEF)
"""
struct TwoPointBVProblem{iip} end # The iip is needed to make type stable construction easier

@doc doc"""

Defines an BVP problem.
Documentation Page: https://docs.sciml.ai/DiffEqDocs/stable/types/bvp_types/

## Mathematical Specification of a BVP Problem

To define a BVP Problem, you simply need to give the function ``f`` and the initial
condition ``u_0`` which define an ODE:

```math
\frac{du}{dt} = f(u,p,t)
```

along with an implicit function `bc` which defines the residual equation, where

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
TwoPointBVProblem{isinplace}(f,bc,u0,tspan,p=NullParameters();kwargs...)
BVProblem{isinplace}(f,bc,u0,tspan,p=NullParameters();kwargs...)
```

or if we have an initial guess function `initialGuess(t)` for the given BVP,
we can pass the initial guess to the problem constructors:

```julia
TwoPointBVProblem{isinplace}(f,bc,initialGuess,tspan,p=NullParameters();kwargs...)
BVProblem{isinplace}(f,bc,initialGuess,tspan,p=NullParameters();kwargs...)
```

For any BVP problem type, `bc` must be inplace if `f` is inplace. Otherwise it must be
out-of-place.

If the bvp is a StandardBVProblem (also known as a Multi-Point BV Problem) it must define
either of the following functions

```julia
bc!(residual, u, p, t)
residual = bc(u, p, t)
```

where `residual` computed from the current `u`. `u` is an array of solution values
where `u[i]` is at time `t[i]`, while `p` are the parameters. For a `TwoPointBVProblem`,
`t = tspan`. For the more general `BVProblem`, `u` can be all of the internal
time points, and for shooting type methods `u=sol` the ODE solution.
Note that all features of the `ODESolution` are present in this form.
In both cases, the size of the residual matches the size of the initial condition.

If the bvp is a TwoPointBVProblem then `bc` must be a Tuple `(bca, bcb)` and each of them
must define either of the following functions:

```julia
begin
    bca!(resid_a, u_a, p)
    bcb!(resid_b, u_b, p)
end
begin
    resid_a = bca(u_a, p)
    resid_b = bcb(u_b, p)
end
```

where `resid_a` and `resid_b` are the residuals at the two endpoints, `u_a` and `u_b` are
the solution values at the two endpoints, and `p` are the parameters.

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
struct BVProblem{uType, tType, isinplace, P, F, PT, K} <:
       AbstractBVProblem{uType, tType, isinplace}
    f::F
    u0::uType
    tspan::tType
    p::P
    problem_type::PT
    kwargs::K

    @add_kwonly function BVProblem{iip}(f::AbstractBVPFunction{iip, TP}, u0, tspan,
        p = NullParameters(); problem_type=nothing, kwargs...) where {iip, TP}
        _u0 = prepare_initial_state(u0)
        _tspan = promote_tspan(tspan)
        warn_paramtype(p)
        prob_type = TP ? TwoPointBVProblem{iip}() : StandardBVProblem()
        # Needed to ensure that `problem_type` doesn't get passed in kwargs
        if problem_type === nothing
            problem_type = prob_type
        else
            @assert prob_type === problem_type "This indicates incorrect problem type specification! Users should never pass in `problem_type` kwarg, this exists exclusively for internal use."
        end
        return new{typeof(_u0), typeof(_tspan), iip, typeof(p), typeof(f),
            typeof(problem_type), typeof(kwargs)}(f, _u0, _tspan, p, problem_type, kwargs)
    end

    function BVProblem{iip}(f, bc, u0, tspan, p = NullParameters(); kwargs...) where {iip}
        BVProblem(BVPFunction{iip}(f, bc), u0, tspan, p; kwargs...)
    end
end

TruncatedStacktraces.@truncate_stacktrace BVProblem 3 1 2

function BVProblem(f, bc, u0, tspan, p = NullParameters(); kwargs...)
    iip = isinplace(f, 4)
    return BVProblem{iip}(BVPFunction{iip}(f, bc), u0, tspan, p; kwargs...)
end

function BVProblem(f::AbstractBVPFunction, u0, tspan, p = NullParameters(); kwargs...)
    return BVProblem{isinplace(f)}(f, u0, tspan, p; kwargs...)
end

# This is mostly a fake stuct and isn't used anywhere
# But we need it for function calls like TwoPointBVProblem{iip}(...) = ...
struct TwoPointBVPFunction{iip} end

@inline function TwoPointBVPFunction(args...; kwargs...)
    return BVPFunction(args...; kwargs..., twopoint = Val(true))
end
@inline function TwoPointBVPFunction{iip}(args...; kwargs...) where {iip}
    return BVPFunction{iip}(args...; kwargs..., twopoint = Val(true))
end

function TwoPointBVProblem{iip}(f, bc, u0, tspan, p = NullParameters();
    bcresid_prototype=nothing, kwargs...) where {iip}
    return TwoPointBVProblem(TwoPointBVPFunction{iip}(f, bc; bcresid_prototype), u0, tspan,
        p; kwargs...)
end
function TwoPointBVProblem(f, bc, u0, tspan, p = NullParameters();
    bcresid_prototype=nothing, kwargs...)
    return TwoPointBVProblem(TwoPointBVPFunction(f, bc; bcresid_prototype), u0, tspan, p;
        kwargs...)
end
function TwoPointBVProblem{iip}(f::AbstractBVPFunction{iip, twopoint}, u0, tspan,
    p = NullParameters(); kwargs...) where {iip, twopoint}
    @assert twopoint "`TwoPointBVProblem` can only be used with a `TwoPointBVPFunction`. Instead of using `BVPFunction`, use `TwoPointBVPFunction` or pass a kwarg `twopoint=Val(true)` during the construction of the `BVPFunction`."
    return BVProblem{iip}(f, u0, tspan, p; kwargs...)
end
function TwoPointBVProblem(f::AbstractBVPFunction{iip, twopoint}, u0, tspan,
    p = NullParameters(); kwargs...) where {iip, twopoint}
    @assert twopoint "`TwoPointBVProblem` can only be used with a `TwoPointBVPFunction`. Instead of using `BVPFunction`, use `TwoPointBVPFunction` or pass a kwarg `twopoint=Val(true)` during the construction of the `BVPFunction`."
    return BVProblem{iip}(f, u0, tspan, p; kwargs...)
end

# Allow previous timeseries solution
function TwoPointBVProblem(f::AbstractODEFunction, bc, sol::T, tspan::Tuple,
    p = NullParameters(); kwargs...) where {T <: AbstractTimeseriesSolution}
    return TwoPointBVProblem(f, bc, sol.u, tspan, p; kwargs...)
end
# Allow initial guess function for the initial guess
function TwoPointBVProblem(f::AbstractODEFunction, bc, initialGuess, tspan::AbstractVector,
    p = NullParameters(); kwargs...)
    u0 = [initialGuess(i) for i in tspan]
    return TwoPointBVProblem(f, bc, u0, (tspan[1], tspan[end]), p; kwargs...)
end
