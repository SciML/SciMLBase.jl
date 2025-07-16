abstract type AbstractClock end

@data Clocks<:AbstractClock begin
    ContinuousClock
    struct PeriodicClock
        dt::Union{Nothing, Float64, Rational{Int}}
        phase::Float64 = 0.0
    end
    SolverStepClock
    struct EventClock
        id::Symbol
    end
end

@derive Clocks[Show, Hash, Eq]

# for backwards compatibility
const TimeDomain = AbstractClock
using .Clocks: ContinuousClock, PeriodicClock, SolverStepClock, EventClock
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

isclock(c::Clocks.Type) = @match c begin
    PeriodicClock() => true
    _ => false
end
isclock(::TimeDomain) = false

issolverstepclock(c::Clocks.Type) = @match c begin
    SolverStepClock() => true
    _ => false
end
issolverstepclock(::TimeDomain) = false

iscontinuous(c::Clocks.Type) = @match c begin
    ContinuousClock() => true
    _ => false
end
iscontinuous(::TimeDomain) = false

iseventclock(c::Clocks.Type) = @match c begin
    EventClock() => true
    _ => false
end
iseventclock(::TimeDomain) = false

is_discrete_time_domain(c::TimeDomain) = !iscontinuous(c)

# workaround for https://github.com/Roger-luo/Moshi.jl/issues/43
isclock(::Any) = false
issolverstepclock(::Any) = false
iscontinuous(::Any) = false
iseventclock(::Any) = false
is_discrete_time_domain(::Any) = false

# public
function first_clock_tick_time(c::Clocks.Type, t0)
    @match c begin
        PeriodicClock(dt) => ceil(t0 / dt) * dt
        SolverStepClock() => t0
        ContinuousClock() => error("ContinuousClock() is not a discrete clock")
        EventClock() => error("Event clocks do not have a defined first tick time.")
        _ => error("Unimplemented for clock $c")
    end
end

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

    return @match c begin
        PeriodicClock(dt) => ceil(sol.prob.tspan[1] / dt) * dt .+ (ic.idx .- 1) .* dt
        SolverStepClock() => begin
            ssc_idx = findfirst(eachindex(sol.discretes)) do i
                !isa(sol.discretes[i].t, AbstractRange)
            end
            sol.discretes[ssc_idx].t[ic.idx]
        end
        ContinuousClock() => sol.t[ic.idx]
    end
end
