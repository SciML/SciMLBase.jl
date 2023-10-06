const DISCRETE_INPLACE_DEFAULT = DiscreteFunction{true}((du, u, p, t) -> du .= u)
const DISCRETE_OUTOFPLACE_DEFAULT = DiscreteFunction{false}((u, p, t) -> u)

@doc doc"""

Defines a discrete dynamical system problem.
Documentation Page: https://docs.sciml.ai/DiffEqDocs/stable/types/discrete_types/

## Mathematical Specification of a Discrete Problem

To define a Discrete Problem, you simply need to give the function ``f`` and the initial
condition ``u_0`` which define a function map:

```math
u_{n+1} = f(u_{n},p,t_{n+1})
```

`f` should be specified as `f(un,p,t)` (or in-place as `f(unp1,un,p,t)`), and `u_0` should
be an AbstractArray (or number) whose geometry matches the desired geometry of `u`.
Note that we are not limited to numbers or vectors for `u₀`; one is allowed to
provide `u₀` as arbitrary matrices / higher dimension tensors as well. ``u_{n+1}`` only depends on the previous
iteration ``u_{n}`` and ``t_{n+1}``. The default ``t_{n+1}`` of `FunctionMap` is
``t_n = t_0 + n*dt`` (with `dt=1` being the default). For continuous-time Markov chains,
this is the time at which the change is occurring.

Note that if the discrete solver is set to have `scale_by_time=true`, then the problem
is interpreted as the map:

```math
u_{n+1} = u_n + dt f(u_{n},p,t_{n+1})
```

## Problem Type

### Constructors

- `DiscreteProblem(f::ODEFunction,u0,tspan,p=NullParameters();kwargs...)` :
  Defines the discrete problem with the specified functions.
- `DiscreteProblem{isinplace,specialize}(f,u0,tspan,p=NullParameters();kwargs...)` :
  Defines the discrete problem with the specified functions.
- `DiscreteProblem{isinplace,specialize}(u0,tspan,p=NullParameters();kwargs...)` :
  Defines the discrete problem with the identity map.

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

For specifying Jacobians and mass matrices, see the
[DiffEqFunctions](@ref performance_overloads)
page.

### Fields

* `f`: The function in the map.
* `u0`: The initial condition.
* `tspan`: The timespan for the problem.
* `p`: The parameters for the problem. Defaults to `NullParameters`
* `kwargs`: The keyword arguments passed onto the solves.

#### Note About Timing

Note that if no `dt` and not `tstops` is given, it's assumed that `dt=1` and thus
`tspan=(0,n)` will solve for `n` iterations. If in the solver `dt` is given, then
the number of iterations will change. And if `tstops` is not empty, the solver will
revert to the standard behavior of fixed timestep methods, which is "step to each
tstop".
"""
struct DiscreteProblem{uType, tType, isinplace, P, F, K} <:
       AbstractDiscreteProblem{uType, tType, isinplace}
    """The function in the map."""
    f::F
    """The initial condition."""
    u0::uType
    """The timespan for the problem."""
    tspan::tType
    """The parameter values of the function."""
    p::P
    """ A callback to be applied to every solver which uses the problem."""
    kwargs::K
    @add_kwonly function DiscreteProblem{iip}(f::AbstractDiscreteFunction{iip},
        u0, tspan::Tuple, p = NullParameters();
        kwargs...) where {iip}
        _u0 = prepare_initial_state(u0)
        _tspan = promote_tspan(tspan)
        warn_paramtype(p)
        new{typeof(_u0), typeof(_tspan), isinplace(f, 4),
            typeof(p),
            typeof(f), typeof(kwargs)}(f,
            _u0,
            _tspan,
            p,
            kwargs)
    end

    function DiscreteProblem{iip}(u0::Nothing, tspan::Nothing, p = NullParameters();
        callback = nothing) where {iip}
        if iip
            f = DISCRETE_INPLACE_DEFAULT
        else
            f = DISCRETE_OUTOFPLACE_DEFAULT
        end
        new{Nothing, Nothing, iip, typeof(p),
            typeof(f), typeof(callback)}(f,
            nothing,
            nothing,
            p,
            callback)
    end

    function DiscreteProblem{iip}(f, u0, tspan, p = NullParameters(); kwargs...) where {iip}
        DiscreteProblem(DiscreteFunction{iip}(f), u0, tspan, p; kwargs...)
    end
end

TruncatedStacktraces.@truncate_stacktrace DiscreteProblem 3 1 2

"""
    DiscreteProblem{isinplace}(f,u0,tspan,p=NullParameters(),callback=nothing)

Defines a discrete problem with the specified functions.
"""
function DiscreteProblem(f::AbstractDiscreteFunction, u0, tspan::Tuple,
    p = NullParameters(); kwargs...)
    DiscreteProblem{isinplace(f)}(f, u0, tspan, p; kwargs...)
end

function DiscreteProblem(f::Base.Callable, u0, tspan::Tuple, p = NullParameters();
    kwargs...)
    iip = isinplace(f, 4)
    DiscreteProblem(DiscreteFunction{iip}(f), u0, tspan, p; kwargs...)
end

"""
$(SIGNATURES)

Define a discrete problem with the identity map.
"""
function DiscreteProblem(u0::Union{AbstractArray, Number}, tspan::Tuple,
    p = NullParameters(); kwargs...)
    iip = typeof(u0) <: AbstractArray
    if iip
        f = DISCRETE_INPLACE_DEFAULT
    else
        f = DISCRETE_OUTOFPLACE_DEFAULT
    end
    DiscreteProblem(f, u0, tspan, p; kwargs...)
end
