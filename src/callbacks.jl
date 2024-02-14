# Necessary to have initialize set u_modified to false if all don't do anything
# otherwise unnecessary save
INITIALIZE_DEFAULT(cb, u, t, integrator) = u_modified!(integrator, false)
FINALIZE_DEFAULT(cb, u, t, integrator) = nothing

@enum RootfindOpt::Int8 begin
    NoRootFind = 0
    LeftRootFind = 1
    RightRootFind = 2
end

function Base.convert(::Type{RootfindOpt}, b::Bool)
    return b ? LeftRootFind : NoRootFind
end

"""
```julia
ContinuousCallback(condition,affect!,affect_neg!;
                   initialize = INITIALIZE_DEFAULT,
                   finalize = FINALIZE_DEFAULT,
                   idxs = nothing,
                   rootfind=LeftRootFind,
                   save_positions=(true,true),
                   interp_points=10,
                   abstol=10eps(),reltol=0,repeat_nudge=1//100)
```

```julia
function ContinuousCallback(condition,affect!;
                   initialize = INITIALIZE_DEFAULT,
                   finalize = FINALIZE_DEFAULT,
                   idxs = nothing,
                   rootfind=LeftRootFind,
                   save_positions=(true,true),
                   affect_neg! = affect!,
                   interp_points=10,
                   abstol=10eps(),reltol=0,repeat_nudge=1//100)
```

Contains a single callback whose `condition` is a continuous function. The callback is triggered when this function evaluates to 0.

# Arguments
- `condition`: This is a function `condition(u,t,integrator)` for declaring when
  the callback should be used. A callback is initiated if the condition hits
  `0` within the time interval. See the [Integrator Interface](@ref integrator) documentation for information about `integrator`.
- `affect!`: This is the function `affect!(integrator)` where one is allowed to
  modify the current state of the integrator. If you do not pass an `affect_neg!`
  function, it is called when `condition` is found to be `0` (at a root) and
  the cross is either an upcrossing (from negative to positive) or a downcrossing
  (from positive to negative). You need to explicitly pass `nothing` as the
  `affect_neg!` argument if it should only be called at upcrossings, e.g.
  `ContinuousCallback(condition, affect!, nothing)`. For more information on what can
  be done, see the [Integrator Interface](@ref integrator) manual page. Modifications to
  `u` are safe in this function.
- `affect_neg!=affect!`: This is the function `affect_neg!(integrator)` where one is allowed to
  modify the current state of the integrator. This is called when `condition` is
  found to be `0` (at a root) and the cross is a downcrossing (from positive to
  negative). For more information on what can
  be done, see the [Integrator Interface](@ref integrator) manual page. Modifications to
  `u` are safe in this function.
- `rootfind=LeftRootFind`: This is a flag to specify the type of rootfinding to do for finding
  event location. If this is set to `LeftRootfind`, the solution will be backtracked to the point where
  `condition==0` and if the solution isn't exact, the left limit of root is used. If set to
  `RightRootFind`, the solution would be set to the right limit of the root. Otherwise, the systems and
  the `affect!` will occur at `t+dt`. Note that these enums are not exported, and thus one needs to
  reference them as `SciMLBase.LeftRootFind`, `SciMLBase.RightRootFind`, or `SciMLBase.NoRootFind`.
- `interp_points=10`: The number of interpolated points to check the condition. The
  condition is found by checking whether any interpolation point / endpoint has
  a different sign. If `interp_points=0`, then conditions will only be noticed if
  the sign of `condition` is different at `t` than at `t+dt`. This behavior is not
  robust when the solution is oscillatory, and thus it's recommended that one use
  some interpolation points (they're cheap to compute!).
  `0` within the time interval.
- `save_positions=(true,true)`: Boolean tuple for whether to save before and after the `affect!`.
  This saving will occur just before and after the event, only at event times, and
  does not depend on options like `saveat`, `save_everystep`, etc. (i.e. if
  `saveat=[1.0,2.0,3.0]`, this can still add a save point at `2.1` if true).
  For discontinuous changes like a modification to `u` to be
  handled correctly (without error), one should set `save_positions=(true,true)`.
- `idxs=nothing`: The components which will be interpolated into the condition. Defaults
  to `nothing` which means `u` will be all components.
- `initialize`: This is a function `(c,u,t,integrator)` which can be used to initialize
  the state of the callback `c`. It should modify the argument `c` and the return is
  ignored.
- `finalize`: This is a function `(c,u,t,integrator)` which can be used to finalize
  the state of the callback `c`. It can modify the argument `c` and the return is ignored.
- `abstol=1e-14` & `reltol=0`: These are used to specify a tolerance from zero for the rootfinder:
  if the starting condition is less than the tolerance from zero, then no root will be detected.
  This is to stop repeat events happening immediately after a rootfinding event.
- `repeat_nudge = 1//100`: This is used to set the next testing point after a
  previously found zero. Defaults to 1//100, which means after a callback, the next
  sign check will take place at t + dt*1//100 instead of at t to avoid repeats.
"""
struct ContinuousCallback{F1, F2, F3, F4, F5, T, T2, T3, I, R} <: AbstractContinuousCallback
    condition::F1
    affect!::F2
    affect_neg!::F3
    initialize::F4
    finalize::F5
    idxs::I
    rootfind::RootfindOpt
    interp_points::Int
    save_positions::BitArray{1}
    dtrelax::R
    abstol::T
    reltol::T2
    repeat_nudge::T3
    function ContinuousCallback(condition::F1, affect!::F2, affect_neg!::F3,
        initialize::F4, finalize::F5, idxs::I, rootfind,
        interp_points, save_positions, dtrelax::R, abstol::T,
        reltol::T2,
        repeat_nudge::T3) where {F1, F2, F3, F4, F5, T, T2, T3, I, R
    }
        _condition = prepare_function(condition)
        new{typeof(_condition), F2, F3, F4, F5, T, T2, T3, I, R}(_condition,
            affect!, affect_neg!,
            initialize, finalize, idxs, rootfind,
            interp_points,
            BitArray(collect(save_positions)),
            dtrelax, abstol, reltol, repeat_nudge)
    end
end

function ContinuousCallback(condition, affect!, affect_neg!;
    initialize = INITIALIZE_DEFAULT,
    finalize = FINALIZE_DEFAULT,
    idxs = nothing,
    rootfind = LeftRootFind,
    save_positions = (true, true),
    interp_points = 10,
    dtrelax = 1,
    abstol = 10eps(), reltol = 0,
    repeat_nudge = 1 // 100)
    ContinuousCallback(condition, affect!, affect_neg!, initialize, finalize,
        idxs,
        rootfind, interp_points,
        save_positions,
        dtrelax, abstol, reltol, repeat_nudge)
end

function ContinuousCallback(condition, affect!;
    initialize = INITIALIZE_DEFAULT,
    finalize = FINALIZE_DEFAULT,
    idxs = nothing,
    rootfind = LeftRootFind,
    save_positions = (true, true),
    affect_neg! = affect!,
    interp_points = 10,
    dtrelax = 1,
    abstol = 10eps(), reltol = 0, repeat_nudge = 1 // 100)
    ContinuousCallback(condition, affect!, affect_neg!, initialize, finalize, idxs,
        rootfind, interp_points,
        collect(save_positions),
        dtrelax, abstol, reltol, repeat_nudge)
end

TruncatedStacktraces.@truncate_stacktrace ContinuousCallback

"""
```julia
VectorContinuousCallback(condition,affect!,affect_neg!,len;
                         initialize = INITIALIZE_DEFAULT,
                         finalize = FINALIZE_DEFAULT,
                         idxs = nothing,
                         rootfind=LeftRootFind,
                         save_positions=(true,true),
                         interp_points=10,
                         abstol=10eps(),reltol=0,repeat_nudge = 1//100)
```

```julia
VectorContinuousCallback(condition,affect!,len;
                   initialize = INITIALIZE_DEFAULT,
                   finalize = FINALIZE_DEFAULT,
                   idxs = nothing,
                   rootfind=LeftRootFind,
                   save_positions=(true,true),
                   affect_neg! = affect!,
                   interp_points=10,
                   abstol=10eps(),reltol=0,repeat_nudge=1//100)
```

This is also a subtype of `AbstractContinuousCallback`. `CallbackSet` is not feasible when you have many callbacks,
as it doesn't scale well. For this reason, we have `VectorContinuousCallback` - it allows you to have a single callback for
multiple events.

# Arguments

- `condition`: This is a function `condition(out, u, t, integrator)` which should save the condition value in the array `out`
   at the right index. Maximum index of `out` should be specified in the `len` property of callback. So, this way you can have
   a chain of `len` events, which would cause the `i`th event to trigger when `out[i] = 0`.
- `affect!`: This is a function `affect!(integrator, event_index)` which lets you modify `integrator` and it tells you about
   which event occurred using `event_idx` i.e. gives you index `i` for which `out[i]` came out to be zero.
- `len`: Number of callbacks chained. This is compulsory to be specified.

Rest of the arguments have the same meaning as in [`ContinuousCallback`](@ref).
"""
struct VectorContinuousCallback{F1, F2, F3, F4, F5, T, T2, T3, I, R} <:
       AbstractContinuousCallback
    condition::F1
    affect!::F2
    affect_neg!::F3
    len::Int
    initialize::F4
    finalize::F5
    idxs::I
    rootfind::RootfindOpt
    interp_points::Int
    save_positions::BitArray{1}
    dtrelax::R
    abstol::T
    reltol::T2
    repeat_nudge::T3
    function VectorContinuousCallback(condition::F1, affect!::F2, affect_neg!::F3, len::Int,
        initialize::F4, finalize::F5, idxs::I, rootfind,
        interp_points, save_positions, dtrelax::R,
        abstol::T, reltol::T2,
        repeat_nudge::T3) where {F1, F2, F3, F4, F5, T, T2,
        T3, I, R}
        _condition = prepare_function(condition)
        new{typeof(_condition), F2, F3, F4, F5, T, T2, T3, I, R}(_condition,
            affect!, affect_neg!, len,
            initialize, finalize, idxs, rootfind,
            interp_points,
            BitArray(collect(save_positions)),
            dtrelax, abstol, reltol, repeat_nudge)
    end
end

function VectorContinuousCallback(condition, affect!, affect_neg!, len;
    initialize = INITIALIZE_DEFAULT,
    finalize = FINALIZE_DEFAULT,
    idxs = nothing,
    rootfind = LeftRootFind,
    save_positions = (true, true),
    interp_points = 10,
    dtrelax = 1,
    abstol = 10eps(), reltol = 0, repeat_nudge = 1 // 100)
    VectorContinuousCallback(condition, affect!, affect_neg!, len,
        initialize, finalize,
        idxs,
        rootfind, interp_points,
        save_positions, dtrelax,
        abstol, reltol, repeat_nudge)
end

function VectorContinuousCallback(condition, affect!, len;
    initialize = INITIALIZE_DEFAULT,
    finalize = FINALIZE_DEFAULT,
    idxs = nothing,
    rootfind = LeftRootFind,
    save_positions = (true, true),
    affect_neg! = affect!,
    interp_points = 10,
    dtrelax = 1,
    abstol = 10eps(), reltol = 0, repeat_nudge = 1 // 100)
    VectorContinuousCallback(condition, affect!, affect_neg!, len, initialize, finalize,
        idxs,
        rootfind, interp_points,
        collect(save_positions),
        dtrelax, abstol, reltol, repeat_nudge)
end

TruncatedStacktraces.@truncate_stacktrace VectorContinuousCallback

"""
```julia
DiscreteCallback(condition,affect!;
                 initialize = INITIALIZE_DEFAULT,
                 finalize = FINALIZE_DEFAULT,
                 save_positions=(true,true))
```

# Arguments

- `condition`: This is a function `condition(u,t,integrator)` for declaring when
  the callback should be used. A callback is initiated if the condition evaluates
  to `true`. See the [Integrator Interface](@ref integrator) documentation for information about `integrator`.
    - `affect!`: This is the function `affect!(integrator)` where one is allowed to
  modify the current state of the integrator. For more information on what can
  be done, see the [Integrator Interface](@ref integrator) manual page.
- `save_positions`: Boolean tuple for whether to save before and after the `affect!`.
  This saving will occur just before and after the event, only at event times, and
  does not depend on options like `saveat`, `save_everystep`, etc. (i.e. if
  `saveat=[1.0,2.0,3.0]`, this can still add a save point at `2.1` if true).
  For discontinuous changes like a modification to `u` to be
  handled correctly (without error), one should set `save_positions=(true,true)`.
- `initialize`: This is a function `(c,u,t,integrator)` which can be used to initialize
  the state of the callback `c`. It should modify the argument `c` and the return is
  ignored.
- `finalize`: This is a function `(c,u,t,integrator)` which can be used to finalize
  the state of the callback `c`. It should can the argument `c` and the return is
  ignored.
"""
struct DiscreteCallback{F1, F2, F3, F4} <: AbstractDiscreteCallback
    condition::F1
    affect!::F2
    initialize::F3
    finalize::F4
    save_positions::BitArray{1}
    function DiscreteCallback(condition::F1, affect!::F2,
        initialize::F3, finalize::F4,
        save_positions) where {F1, F2, F3, F4}
        _condition = prepare_function(condition)
        new{typeof(_condition), F2, F3, F4}(_condition,
            affect!, initialize, finalize,
            BitArray(collect(save_positions)))
    end
end
function DiscreteCallback(condition, affect!;
    initialize = INITIALIZE_DEFAULT, finalize = FINALIZE_DEFAULT,
    save_positions = (true, true))
    DiscreteCallback(condition, affect!, initialize, finalize, save_positions)
end

TruncatedStacktraces.@truncate_stacktrace DiscreteCallback

"""
$(TYPEDEF)

Multiple callbacks can be chained together to form a `CallbackSet`. A `CallbackSet`
is constructed by passing the constructor `ContinuousCallback`, `DiscreteCallback`,
`VectorContinuousCallback` or other `CallbackSet` instances:

    CallbackSet(cb1,cb2,cb3)

You can pass as many callbacks as you like. When the solvers encounter multiple
callbacks, the following rules apply:

* `ContinuousCallback`s and `VectorContinuousCallback`s are applied before `DiscreteCallback`s. (This is because
  they often implement event-finding that will backtrack the timestep to smaller
  than `dt`).
* For `ContinuousCallback`s and `VectorContinuousCallback`s, the event times are found by rootfinding and only
  the first `ContinuousCallback` or `VectorContinuousCallback` affect is applied.
* The `DiscreteCallback`s are then applied in order. Note that the ordering only
  matters for the conditions: if a previous callback modifies `u` in such a way
  that the next callback no longer evaluates condition to `true`, its `affect`
  will not be applied.

"""
struct CallbackSet{T1 <: Tuple, T2 <: Tuple} <: DECallback
    continuous_callbacks::T1
    discrete_callbacks::T2
end

CallbackSet(callback::AbstractDiscreteCallback) = CallbackSet((), (callback,))
CallbackSet(callback::AbstractContinuousCallback) = CallbackSet((callback,), ())
CallbackSet() = CallbackSet((), ())
CallbackSet(cb::Nothing) = CallbackSet()

# For Varargs, use recursion to make it type-stable
function CallbackSet(callbacks::Union{DECallback, Nothing}...)
    CallbackSet(split_callbacks((), (), callbacks...)...)
end

"""
    split_callbacks(cs, ds, args...)

Split comma separated callbacks into sets of continuous and discrete callbacks.
"""
@inline split_callbacks(cs, ds) = cs, ds
@inline split_callbacks(cs, ds, c::Nothing, args...) = split_callbacks(cs, ds, args...)
@inline function split_callbacks(cs, ds, c::AbstractContinuousCallback, args...)
    split_callbacks((cs..., c), ds, args...)
end
@inline function split_callbacks(cs, ds, d::AbstractDiscreteCallback, args...)
    split_callbacks(cs, (ds..., d), args...)
end
@inline function split_callbacks(cs, ds, d::CallbackSet, args...)
    split_callbacks((cs..., d.continuous_callbacks...), (ds..., d.discrete_callbacks...),
        args...)
end
