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
Documentation Page: [https://docs.sciml.ai/DiffEqDocs/stable/types/bvp_types/](https://docs.sciml.ai/DiffEqDocs/stable/types/bvp_types/)

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
TwoPointBVProblem{isinplace}(f, bc, u0, tspan, p=NullParameters(); kwargs...)
BVProblem{isinplace}(f, bc, u0, tspan, p=NullParameters(); kwargs...)
```

or if we have an initial guess function `initialGuess(p, t)` for the given BVP,
we can pass the initial guess to the problem constructors:

```julia
TwoPointBVProblem{isinplace}(f, bc, initialGuess, tspan, p=NullParameters(); kwargs...)
BVProblem{isinplace}(f, bc, initialGuess, tspan, p=NullParameters(); kwargs...)
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

### Special Keyword Arguments

- `nlls`: Specify that the BVP is a nonlinear least squares problem. Use `Val(true)` or
  `Val(false)` for type stability. By default this is automatically inferred based on the
  size of the input and outputs, however this is type unstable for any array type that
  doesn't store array size as part of type information. If we can't reliably infer this,
  we set it to `Nothing`. Downstreams solvers must be setup to deal with this case.
"""
struct BVProblem{uType, tType, isinplace, nlls, P, F, PT, K} <:
       AbstractBVProblem{uType, tType, isinplace, nlls}
    f::F
    u0::uType
    tspan::tType
    p::P
    problem_type::PT
    kwargs::K

    @add_kwonly function BVProblem{iip}(f::AbstractBVPFunction{iip, TP}, u0, tspan,
            p = NullParameters(); problem_type = nothing, nlls = nothing,
            kwargs...) where {iip, TP}
        _u0 = prepare_initial_state(u0)
        _tspan = promote_tspan(tspan)
        warn_paramtype(p)
        prob_type = TP ? TwoPointBVProblem{iip}() : StandardBVProblem()

        # Needed to ensure that `problem_type` doesn't get passed in kwargs
        if problem_type === nothing
            problem_type = prob_type
        else
            @assert prob_type===problem_type "This indicates incorrect problem type specification! Users should never pass in `problem_type` kwarg, this exists exclusively for internal use."
        end

        if nlls === nothing
            if !hasmethod(length, Tuple{typeof(_u0)})
                # If _u0 is a function for initial guess we won't be able to infer
                __u0 = _u0 isa Function ?
                       (hasmethod(_u0, Tuple{typeof(p), typeof(first(_tspan))}) ?
                        _u0(p, first(_tspan)) : _u0(first(_tspan))) : nothing
            else
                __u0 = _u0
            end
            # Try to infer it
            if __u0 isa Nothing
                _nlls = Nothing
            elseif problem_type isa TwoPointBVProblem
                if f.bcresid_prototype !== nothing
                    l1, l2 = length(f.bcresid_prototype[1]), length(f.bcresid_prototype[2])
                    _nlls = l1 + l2 != length(__u0)
                else
                    # iip without bcresid_prototype is not possible
                    if !iip
                        l1 = length(f.bc[1](u0, p))
                        l2 = length(f.bc[2](u0, p))
                        _nlls = l1 + l2 != length(__u0)
                    end
                end
            else
                if f.bcresid_prototype !== nothing
                    _nlls = length(f.bcresid_prototype) != length(__u0)
                else
                    _nlls = Nothing # Cannot reliably infer
                end
            end
        else
            _nlls = _unwrap_val(nlls)
        end

        return new{typeof(_u0), typeof(_tspan), iip, _nlls, typeof(p), typeof(f),
            typeof(problem_type), typeof(kwargs)}(f, _u0, _tspan, p, problem_type, kwargs)
    end

    function BVProblem{iip}(f, bc, u0, tspan, p = NullParameters(); kwargs...) where {iip}
        BVProblem(BVPFunction{iip}(f, bc), u0, tspan, p; kwargs...)
    end
end

function BVProblem(f, bc, u0, tspan, p = NullParameters(); kwargs...)
    iip = isinplace(f, 4)
    return BVProblem{iip}(BVPFunction{iip}(f, bc), u0, tspan, p; kwargs...)
end

function BVProblem(f::AbstractBVPFunction, u0, tspan, p = NullParameters(); kwargs...)
    return BVProblem{isinplace(f)}(f, u0, tspan, p; kwargs...)
end

# This is mostly a fake struct and isn't used anywhere
# But we need it for function calls like TwoPointBVProblem{iip}(...) = ...
struct TwoPointBVPFunction{iip} end

@inline function TwoPointBVPFunction(args...; kwargs...)
    return BVPFunction(args...; kwargs..., twopoint = Val(true))
end
@inline function TwoPointBVPFunction{iip}(args...; kwargs...) where {iip}
    return BVPFunction{iip}(args...; kwargs..., twopoint = Val(true))
end

function TwoPointBVProblem{iip}(f, bc, u0, tspan, p = NullParameters();
        bcresid_prototype = nothing, kwargs...) where {iip}
    return TwoPointBVProblem(TwoPointBVPFunction{iip}(f, bc; bcresid_prototype), u0, tspan,
        p; kwargs...)
end
function TwoPointBVProblem(f, bc, u0, tspan, p = NullParameters();
        bcresid_prototype = nothing, kwargs...)
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

######################## SecondOrderBVProblem ########################

"""
$(TYPEDEF)
"""
struct StandardSecondOrderBVProblem end

"""
$(TYPEDEF)
"""
struct TwoPointSecondOrderBVProblem{iip} end # The iip is needed to make type stable construction easier

@doc doc"""

Defines a second order BVP problem.
Documentation Page: [https://docs.sciml.ai/DiffEqDocs/stable/types/bvp_types/](https://docs.sciml.ai/DiffEqDocs/stable/types/bvp_types/)

## Mathematical Specification of a second order BVP Problem

To define a second order BVP Problem, you simply need to give the function ``f`` and the initial
condition ``u_0`` which define an ODE:

```math
\frac{ddu}{dt} = f(du,u,p,t)
```

along with an implicit function `bc` which defines the residual equation, where

```math
bc(du,u,p,t) = 0
```

is the manifold on which the solution must live. A common form for this is the
two-point `SecondOrderBVProblem` where the manifold defines the solution at two points:

```math
g(u(t_0),u'(t_0)) = 0
g(u(t_f),u'(t_f)) = 0
```

## Problem Type

### Constructors

```julia
TwoPointSecondOrderBVProblem{isinplace}(f, bc, u0, tspan, p=NullParameters(); kwargs...)
SecondOrderBVProblem{isinplace}(f, bc, u0, tspan, p=NullParameters(); kwargs...)
```

or if we have an initial guess function `initialGuess(p, t)` for the given BVP,
we can pass the initial guess to the problem constructors:

```julia
TwoPointSecondOrderBVProblem{isinplace}(f, bc, initialGuess, tspan, p=NullParameters(); kwargs...)
SecondOrderBVProblem{isinplace}(f, bc, initialGuess, tspan, p=NullParameters(); kwargs...)
```

For any BVP problem type, `bc` must be inplace if `f` is inplace. Otherwise it must be
out-of-place.

If the bvp is a StandardSecondOrderBVProblem (also known as a Multi-Point BV Problem) it must define
either of the following functions

```julia
bc!(residual, du, u, p, t)
residual = bc(du, u, p, t)
```

where `residual` computed from the current `u`. `u` is an array of solution values
where `u[i]` is at time `t[i]`, while `p` are the parameters. For a `TwoPointBVProblem`,
`t = tspan`. For the more general `BVProblem`, `u` can be all of the internal
time points, and for shooting type methods `u=sol` the ODE solution.
Note that all features of the `ODESolution` are present in this form.
In both cases, the size of the residual matches the size of the initial condition.

If the bvp is a `TwoPointSecondOrderBVProblem` then `bc` must be a Tuple `(bca, bcb)` and each of them
must define either of the following functions:

```julia
begin
    bca!(resid_a, du_a, u_a, p)
    bcb!(resid_b, du_b, u_b, p)
end
begin
    resid_a = bca(du_a, u_a, p)
    resid_b = bcb(du_b, u_b, p)
end
```

where `resid_a` and `resid_b` are the residuals at the two endpoints, `u_a` and `u_b` are
the solution values at the two endpoints, `du_a` and `du_b` are the derivative of solution values at the two endpoints, and `p` are the parameters.


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
struct SecondOrderBVProblem{uType, tType, isinplace, nlls, P, F, PT, K} <:
       AbstractBVProblem{uType, tType, isinplace, nlls}
    f::F
    u0::uType
    tspan::tType
    p::P
    problem_type::PT
    kwargs::K

    @add_kwonly function SecondOrderBVProblem{iip}(
            f::DynamicalBVPFunction{iip, specialize, TP}, u0, tspan,
            p = NullParameters(); problem_type = nothing, nlls = nothing,
            kwargs...) where {iip, specialize, TP}
        _u0 = prepare_initial_state(u0)
        _tspan = promote_tspan(tspan)
        warn_paramtype(p)
        prob_type = TP ? TwoPointSecondOrderBVProblem{iip}() : StandardSecondOrderBVProblem()

        # Needed to ensure that `problem_type` doesn't get passed in kwargs
        if problem_type === nothing
            problem_type = prob_type
        else
            @assert prob_type===problem_type "This indicates incorrect problem type specification! Users should never pass in `problem_type` kwarg, this exists exclusively for internal use."
        end
        
        return new{typeof(_u0), typeof(_tspan), iip, typeof(nlls), typeof(p), typeof(f),
            typeof(problem_type), typeof(kwargs)}(f, _u0, _tspan, p, problem_type, kwargs)
    end

    function SecondOrderBVProblem{iip}(
            f, bc, u0, tspan, p = NullParameters(); kwargs...) where {iip}
        SecondOrderBVProblem(DynamicalBVPFunction{iip}(f, bc), u0, tspan, p; kwargs...)
    end
end

function SecondOrderBVProblem(f, bc, u0, tspan, p = NullParameters(); kwargs...)
    iip = isinplace(f, 5)
    return SecondOrderBVProblem{iip}(
        DynamicalBVPFunction{iip}(f, bc), u0, tspan, p; kwargs...)
end

function SecondOrderBVProblem(
        f::DynamicalBVPFunction, u0, tspan, p = NullParameters(); kwargs...)
    return SecondOrderBVProblem{isinplace(f)}(f, u0, tspan, p; kwargs...)
end

# This is mostly a fake struct and isn't used anywhere
# But we need it for function calls like TwoPointBVProblem{iip}(...) = ...
struct TwoPointDynamicalBVPFunction{iip} end

@inline function TwoPointDynamicalBVPFunction(args...; kwargs...)
    return DynamicalBVPFunction(args...; kwargs..., twopoint = Val(true))
end
@inline function TwoPointDynamicalBVPFunction{iip}(args...; kwargs...) where {iip}
    return DynamicalBVPFunction{iip}(args...; kwargs..., twopoint = Val(true))
end

function TwoPointSecondOrderBVProblem{iip}(f, bc, u0, tspan, p = NullParameters();
        bcresid_prototype = nothing, kwargs...) where {iip}
    return TwoPointSecondOrderBVProblem(
        TwoPointDynamicalBVPFunction{iip}(f, bc; bcresid_prototype), u0, tspan,
        p; kwargs...)
end
function TwoPointSecondOrderBVProblem(f, bc, u0, tspan, p = NullParameters();
        bcresid_prototype = nothing, kwargs...)
    return TwoPointSecondOrderBVProblem(
        TwoPointDynamicalBVPFunction(f, bc; bcresid_prototype), u0, tspan, p;
        kwargs...)
end
function TwoPointSecondOrderBVProblem{iip}(
        f::AbstractBVPFunction{iip, twopoint}, u0, tspan,
        p = NullParameters(); kwargs...) where {iip, twopoint}
    @assert twopoint "`TwoPointSecondOrderBVProblem` can only be used with a `TwoPointDynamicalBVPFunction`. Instead of using `DynamicalBVPFunction`, use `TwoPointDynamicalBVPFunction` or pass a kwarg `twopoint=Val(true)` during the construction of the `DynamicalBVPFunction`."
    return SecondOrderBVProblem{iip}(f, u0, tspan, p; kwargs...)
end
function TwoPointSecondOrderBVProblem(f::AbstractBVPFunction{iip, twopoint}, u0, tspan,
        p = NullParameters(); kwargs...) where {iip, twopoint}
    @assert twopoint "`TwoPointSecondOrderBVProblem` can only be used with a `TwoPointDynamicalBVPFunction`. Instead of using `DynamicalBVPFunction`, use `TwoPointDynamicalBVPFunction` or pass a kwarg `twopoint=Val(true)` during the construction of the `DynamicalBVPFunction`."
    return SecondOrderBVProblem{iip}(f, u0, tspan, p; kwargs...)
end

# Allow previous timeseries solution
function TwoPointSecondOrderBVProblem(f::AbstractODEFunction, bc, sol::T, tspan::Tuple,
        p = NullParameters(); kwargs...) where {T <: AbstractTimeseriesSolution}
    return TwoPointSecondOrderBVProblem(f, bc, sol.u, tspan, p; kwargs...)
end
# Allow initial guess function for the initial guess
function TwoPointSecondOrderBVProblem(
        f::AbstractODEFunction, bc, initialGuess, tspan::AbstractVector,
        p = NullParameters(); kwargs...)
    u0 = [initialGuess(i) for i in tspan]
    return TwoPointSecondOrderBVProblem(f, bc, u0, (tspan[1], tspan[end]), p; kwargs...)
end
