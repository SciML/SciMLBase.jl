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
phase offset.

# Fields

$(TYPEDFIELDS)
"""
struct PeriodicClock <: AbstractClock
    """
    Nominal tick interval, or `nothing` when the interval is left for downstream
    inference.
    """
    dt::Union{Nothing, Float64, Rational{Int}}
    """
    Phase offset metadata for the periodic clock.
    """
    phase::Float64
end
PeriodicClock(dt; phase = 0.0) = PeriodicClock(dt, Float64(phase))
PeriodicClock(; dt = nothing, phase = 0.0) = PeriodicClock(dt, Float64(phase))

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

Namespace for clock marker types used by hybrid and clocked SciML systems.
The module exports `ContinuousClock`, `PeriodicClock`, `SolverStepClock`, and
`EventClock` for backwards-compatible qualified access such as
`SciMLBase.Clocks.PeriodicClock`.
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
numeric intervals are converted to `Float64`. Calling `Clock(; phase = phase)`
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

Return the first tick time for a discrete clock at or after `t0`.

For [`PeriodicClock`](@ref), this is the first multiple of `dt` at or after
`t0`. For [`SolverStepClock`](@ref), the first tick is `t0`. Continuous and
event clocks do not have a generic first tick time and throw an error.
"""
first_clock_tick_time(c::PeriodicClock, t0) = ceil(t0 / c.dt) * c.dt
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

# public
"""
    $(TYPEDSIGNATURES)

Return a `SciMLBase.IndexedClock` representing the subset of the time points that the clock
ticked indicated by `idx`.
"""
Base.getindex(c::AbstractClock, idx) = IndexedClock(c, idx)

# public
"""
    $(TYPEDSIGNATURES)

Convert an indexed clock reference into concrete independent-variable values
for a saved time-series solution.

`IndexedClock` stores a clock and one or more tick indices, but the actual tick
times depend on the solve. Periodic clocks are reconstructed from the problem
start time and tick interval, solver-step clocks read the matching discrete
time series stored on the solution, and continuous clocks index directly into
`sol.t`. Unsupported clock types throw an error.
"""
function canonicalize_indexed_clock(ic::IndexedClock, sol::AbstractTimeseriesSolution)
    c = ic.clock
    return if c isa PeriodicClock
        ceil(sol.prob.tspan[1] / c.dt) * c.dt .+ (ic.idx .- 1) .* c.dt
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
