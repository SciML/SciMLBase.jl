abstract type AbstractClock end

struct ContinuousClock <: AbstractClock end
struct PeriodicClock <: AbstractClock
    dt::Union{Nothing, Float64, Rational{Int}}
    phase::Float64
end
PeriodicClock(dt; phase = 0.0) = PeriodicClock(dt, Float64(phase))
PeriodicClock(; dt = nothing, phase = 0.0) = PeriodicClock(dt, Float64(phase))
struct SolverStepClock <: AbstractClock end
struct EventClock <: AbstractClock
    id::Symbol
end

# Namespace module for backwards-compatible Clocks.ContinuousClock etc.
module Clocks
    using ..SciMLBase: AbstractClock, ContinuousClock, PeriodicClock, SolverStepClock, EventClock
    const Type = AbstractClock
    export ContinuousClock, PeriodicClock, SolverStepClock, EventClock
end

# for backwards compatibility
const TimeDomain = AbstractClock
const Continuous = ContinuousClock()
(clock::TimeDomain)() = clock

Base.Broadcast.broadcastable(d::TimeDomain) = Ref(d)

"""
    Clock(dt)
    Clock()

The default periodic clock with tick interval `dt`. If `dt` is left unspecified, it will
be inferred (if possible).
"""
Clock(dt::Union{<:Rational, Float64}; phase = 0.0) = PeriodicClock(dt, phase)
Clock(dt; phase = 0.0) = PeriodicClock(convert(Float64, dt), phase)
Clock(; phase = 0.0) = PeriodicClock(nothing, phase)

@doc """
    SolverStepClock

A clock that ticks at each solver step (sometimes referred to as "continuous sample time").
This clock **does generally not have equidistant tick intervals**, instead, the tick
interval depends on the adaptive step-size selection of the continuous solver, as well as
any continuous event handling. If adaptivity of the solver is turned off and there are no
continuous events, the tick interval will be given by the fixed solver time step `dt`.

Due to possibly non-equidistant tick intervals, this clock should typically not be used with
discrete-time systems that assume a fixed sample time, such as PID controllers and digital
filters.
""" SolverStepClock

"""
    isclock(clock)

Returns `true` if the object is a valid clock type (specifically a `PeriodicClock`).
This function is used for type checking in clock-dependent logic.
"""
isclock(@nospecialize(clk)) = clk isa PeriodicClock

"""
    issolverstepclock(clock)

Returns `true` if the clock is a `SolverStepClock` that triggers at every solver step.
This is useful for monitoring solver progress or implementing step-dependent logic.
"""
issolverstepclock(@nospecialize(clk)) = clk isa SolverStepClock

"""
    iscontinuous(clock)

Returns `true` if the clock operates in continuous time (i.e., is a `ContinuousClock`).
Continuous clocks allow events to occur at any real-valued time instant.
"""
iscontinuous(@nospecialize(clk)) = clk isa ContinuousClock

"""
    iseventclock(clock)

Returns `true` if the clock is an `EventClock` that triggers based on specific events.
Event clocks are used for condition-based triggering in hybrid systems.
"""
iseventclock(@nospecialize(clk)) = clk isa EventClock

"""
    is_discrete_time_domain(clock)

Returns `true` if the clock operates in discrete time (i.e., is not a continuous clock).
Discrete time domains have specific sampling intervals or event-based triggering.
"""
is_discrete_time_domain(::Nothing) = false
is_discrete_time_domain(@nospecialize(clk)) = !iscontinuous(clk)

# public
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

Return the time points in the interval
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
