"""
$(TYPEDEF)

Base interface for time-domain clock objects.

Clocks describe the independent-variable times at which discrete quantities are
sampled or updated. SciMLBase defines periodic clocks, solver-step clocks,
continuous clocks, and event clocks as lightweight descriptors. Downstream
packages may store clocks in symbolic indexing metadata and solution discrete
time-series data so callers can evaluate `sol(clock[idx])` at the corresponding
times.

Clock objects are scalar descriptors for broadcasting, callable as `clock()`
for backwards-compatible code paths, and can be indexed to form an
[`IndexedClock`](@ref). Use the trait helpers [`isclock`](@ref),
[`issolverstepclock`](@ref), [`iscontinuous`](@ref), and
[`is_discrete_time_domain`](@ref) when code needs to branch on clock semantics.
"""
abstract type AbstractClock end

"""
$(TYPEDEF)

Clock representing the continuous independent-variable domain.

`ContinuousClock()` is used when a quantity should be interpreted on the same
continuous time axis as the solution itself. Indexing a continuous clock and
canonicalizing it against a saved time-series solution selects entries from
`sol.t`.
"""
struct ContinuousClock <: AbstractClock end

"""
$(TYPEDEF)

Clock with nominal periodic ticks.

`PeriodicClock(dt; phase = 0.0)` describes ticks separated by `dt`. A `dt` of
`nothing` means the interval has not been fixed and may be inferred by
downstream tooling. `phase` is stored as clock metadata for packages that need a
phase offset. Rational values of `dt` and `phase` are stored as they are, which
allows downstream tooling to compute tick times exactly, for example such that
ticks of two clocks that coincide mathematically are also equal as floating-point
numbers. Other real values are converted to `Float64`. The type parameters `D` and
`P` are the types of `dt` and `phase`.

# Fields

$(TYPEDFIELDS)
"""
struct PeriodicClock{D <: Union{Nothing, Real}, P <: Real} <: AbstractClock
    """
    Nominal tick interval, or `nothing` when the interval is left for downstream
    inference.
    """
    dt::D
    """
    Phase offset metadata for the periodic clock.
    """
    phase::P
    function PeriodicClock(dt, phase)
        dt′, phase′ = _clock_value(dt), _clock_value(phase)
        return new{typeof(dt′), typeof(phase′)}(dt′, phase′)
    end
end
_clock_value(x::Nothing) = x
_clock_value(x::Rational) = x
_clock_value(x) = Float64(x)
# Clocks with numerically equal periods and phases are equal, irrespective of
# whether the values are stored as `Float64` or as rationals.
function Base.:(==)(a::PeriodicClock, b::PeriodicClock)
    return a.dt == b.dt && a.phase == b.phase
end
Base.isequal(a::PeriodicClock, b::PeriodicClock) = isequal(a.dt, b.dt) && isequal(a.phase, b.phase)
Base.hash(c::PeriodicClock, h::UInt) = hash(c.phase, hash(c.dt, hash(PeriodicClock, h)))
PeriodicClock(dt; phase = 0.0) = PeriodicClock(dt, phase)
PeriodicClock(; dt = nothing, phase = 0.0) = PeriodicClock(dt, phase)

"""
$(TYPEDEF)

Clock that ticks at accepted solver steps.

Solver-step clocks do not generally have equidistant ticks: adaptive step-size
selection and event handling can change the tick times. They are useful for
querying quantities saved on the solver's internal step sequence, but they are
not a fixed-sample-rate clock unless the solver itself is fixed-step and has no
events that alter the step sequence.
"""
struct SolverStepClock <: AbstractClock end

"""
$(TYPEDEF)

Clock identified by a named event.

`EventClock(id)` is a descriptor for event-triggered discrete time domains. The
event identity is stored in `id`; concrete event detection and storage semantics
are supplied by downstream packages that attach event-clock data to a solution
or symbolic system.

# Fields

$(TYPEDFIELDS)
"""
struct EventClock <: AbstractClock
    """
    Symbol identifying the event stream associated with this clock.
    """
    id::Symbol
end

"""
    Clocks

Backwards-compatible namespace for `ContinuousClock`, `PeriodicClock`,
`SolverStepClock`, and `EventClock`. New code can import these public clock
types directly from `SciMLBase`; the namespace remains available for code that
uses names such as `SciMLBase.Clocks.PeriodicClock`.
"""
module Clocks
    using ..SciMLBase: AbstractClock, ContinuousClock, PeriodicClock, SolverStepClock, EventClock
    const Type = AbstractClock
    export ContinuousClock, PeriodicClock, SolverStepClock, EventClock
end

# for backwards compatibility
"""
    TimeDomain

Backwards-compatible alias for [`AbstractClock`](@ref).

Use `TimeDomain` in old code paths that dispatch on clock-like time-domain
descriptors. New interface documentation should refer to `AbstractClock` and the
concrete clock types directly.
"""
const TimeDomain = AbstractClock

"""
    Continuous

Singleton [`ContinuousClock`](@ref) value for the continuous time domain.
"""
const Continuous = ContinuousClock()
(clock::TimeDomain)() = clock

Base.Broadcast.broadcastable(d::TimeDomain) = Ref(d)

"""
    Clock(dt)
    Clock(; phase = 0.0)

Construct the default [`PeriodicClock`](@ref).

`Clock(dt; phase = 0.0)` converts numeric `dt` values to a periodic clock with
that tick interval. Rational and `Float64` intervals are preserved, while other
numeric intervals are converted to `Float64`. Calling `Clock(; phase)`
leaves `dt` as `nothing`, allowing downstream tooling to infer the interval when
possible.
"""
Clock(dt::Union{<:Rational, Float64}; phase = 0.0) = PeriodicClock(dt, phase)
Clock(dt; phase = 0.0) = PeriodicClock(convert(Float64, dt), phase)
Clock(; phase = 0.0) = PeriodicClock(nothing, phase)

"""
    isclock(clock)

Return `true` when `clock` is a periodic sampled clock.

This legacy trait currently recognizes [`PeriodicClock`](@ref) values only. Use
[`iscontinuous`](@ref), [`issolverstepclock`](@ref), and
[`is_discrete_time_domain`](@ref) for the broader clock-family predicates.
"""
isclock(@nospecialize(clk)) = clk isa PeriodicClock

"""
    issolverstepclock(clock)

Return `true` if `clock` is a [`SolverStepClock`](@ref).
"""
issolverstepclock(@nospecialize(clk)) = clk isa SolverStepClock

"""
    iscontinuous(clock)

Return `true` if `clock` is a [`ContinuousClock`](@ref).
"""
iscontinuous(@nospecialize(clk)) = clk isa ContinuousClock

"""
    iseventclock(clock)

Return `true` if `clock` is an [`EventClock`](@ref).
"""
iseventclock(@nospecialize(clk)) = clk isa EventClock

"""
    is_discrete_time_domain(clock)

Return `true` when `clock` represents a discrete time domain.

`nothing` is treated as not discrete. Any clock that is not continuous is treated
as discrete, including periodic, solver-step, and event clocks.
"""
is_discrete_time_domain(::Nothing) = false
is_discrete_time_domain(@nospecialize(clk)) = !iscontinuous(clk)

# public
"""
    first_clock_tick_time(clock, t0)

Return the first tick time of a discrete clock in a simulation that starts at `t0`.

A [`PeriodicClock`](@ref) ticks at `t0 + phase + k * dt` for integers `k ≥ 0`, so
its first tick is at `t0 + phase`. When `dt` is rational, the offset
`phase + k * dt` is computed in exact arithmetic and rounded once. For [`SolverStepClock`](@ref), the first tick is `t0`. Continuous and event
clocks do not have a generic first tick time and throw an error.
"""
first_clock_tick_time(c::PeriodicClock, t0) = _periodic_tick_time(c, t0, 0)

# Time of the tick with index `k` (counted from zero) of a periodic clock in a
# simulation that starts at `t0`. The arithmetic is that of `PeriodicCallback` of
# DiffEqCallbacks.jl, such that the returned times are equal to the times at which
# the callback fires: for a rational `dt`, the offset `phase + k * dt` is computed
# exactly (a `Float64` phase converted exactly to `Rational{BigInt}`), rounded
# once and added to `t0`. A clock with zero phase ticks at `t0` itself.
function _periodic_tick_time(c::PeriodicClock, t0, k)
    T = typeof(float(t0))
    dt, phase = c.dt, c.phase
    k == 0 && iszero(phase) && return convert(T, t0)
    return if dt isa Rational
        exact_phase = phase isa Rational ? phase : iszero(phase) ? zero(dt) : Rational{BigInt}(phase)
        t0 + convert(T, exact_phase + k * dt)
    else
        convert(T, convert(typeof(dt), t0 + phase) + k * dt)
    end
end

first_clock_tick_time(::SolverStepClock, t0) = t0
first_clock_tick_time(::ContinuousClock, _) = error("ContinuousClock() is not a discrete clock")
first_clock_tick_time(::EventClock, _) = error("Event clocks do not have a defined first tick time.")
function first_clock_tick_time(c::TimeDomain, _)
    error("Unimplemented for clock $c")
end

# public
"""
    $(TYPEDEF)

A struct representing the operation of indexing a clock to obtain a subset of the time
points at which it ticked. The actual list of time points depends on the tick instances
on which the clock was ticking, and can be obtained via `canonicalize_indexed_clock`
by providing a timeseries solution object.

For example, `IndexedClock(PeriodicClock(0.1), 3)` refers to the third time that
`PeriodicClock(0.1)` ticked. If the simulation started at `t = 0`, then this would be
`t = 0.2`. Similarly, `IndexedClock(PeriodicClock(0.1), [1, 5])` refers to `t = 0.0`
and `t = 0.4` in this context.

# Fields

$(TYPEDFIELDS)
"""
struct IndexedClock{C <: AbstractClock, I}
    """
    The clock being indexed. A subtype of `SciMLBase.AbstractClock`
    """
    clock::C
    """
    The subset of indexes being referred to. This can be an integer, an array of integers,
    a range or `Colon()` to refer to all the points that the clock ticked.
    """
    idx::I
end

"""
    Base.getindex(clock::AbstractClock, idx) -> IndexedClock

Select one or more tick indices from `clock` without resolving them to times.

# Arguments

- `clock`: A SciML clock whose ticks will be selected.
- `idx`: An integer, integer collection, range, or `Colon()` selector.

# Returns

- [`IndexedClock`](@ref): A lazy clock/index pair. A solution resolves the pair to
  concrete independent-variable values when it is interpolated.

# Example

```julia
indexed_clock = Clock(0.1)[2:4]
indexed_clock.idx
```
"""
Base.getindex(c::AbstractClock, idx) = IndexedClock(c, idx)

# public
"""
    $(TYPEDSIGNATURES)

Convert an indexed clock reference into concrete independent-variable values
for a saved time-series solution.

`IndexedClock` stores a clock and one or more tick indices, but the actual tick
times depend on the solve. Periodic clocks are reconstructed from the problem
start time, tick interval and phase (the clock ticks at
`tspan[1] + phase + (idx - 1) * dt`), solver-step clocks read the matching discrete
time series stored on the solution, and continuous clocks index directly into
`sol.t`. Unsupported clock types throw an error.
"""
function canonicalize_indexed_clock(ic::IndexedClock, sol::AbstractTimeseriesSolution)
    c = ic.clock
    return if c isa PeriodicClock
        _periodic_tick_time.(Ref(c), sol.prob.tspan[1], ic.idx .- 1)
    elseif c isa SolverStepClock
        ssc_idx = findfirst(eachindex(sol.discretes)) do i
            !isa(sol.discretes[i].t, AbstractRange)
        end
        sol.discretes[ssc_idx].t[ic.idx]
    elseif c isa ContinuousClock
        sol.t[ic.idx]
    else
        error("Unimplemented for clock $c")
    end
end
