# Fallback clock implementations when Moshi is not available
# These provide basic clock functionality without pattern matching

# Simple struct-based clock definitions (fallback when Moshi not loaded)
abstract type TimeDomain end

struct ContinuousClock <: TimeDomain end

struct PeriodicClock <: TimeDomain
    dt::Union{Nothing, Float64, Rational{Int}}
    phase::Float64
    function PeriodicClock(dt::Union{Nothing, Float64, Rational{Int}}, phase::Float64 = 0.0)
        new(dt, phase)
    end
end

struct SolverStepClock <: TimeDomain end

# for backwards compatibility
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

# Fallback implementations without pattern matching
isclock(c::PeriodicClock) = true
isclock(c::TimeDomain) = false

issolverstepclock(c::SolverStepClock) = true
issolverstepclock(c::TimeDomain) = false

iscontinuous(c::ContinuousClock) = true
iscontinuous(c::TimeDomain) = false

is_discrete_time_domain(c::TimeDomain) = !iscontinuous(c)

# workaround for fallback when argument is not a TimeDomain
isclock(::Any) = false
issolverstepclock(::Any) = false
iscontinuous(::Any) = false
is_discrete_time_domain(::Any) = false

function first_clock_tick_time(c::PeriodicClock, t0)
    dt = c.dt
    ceil(t0 / dt) * dt
end
function first_clock_tick_time(c::SolverStepClock, t0)
    t0
end
function first_clock_tick_time(c::ContinuousClock, t0)
    error("ContinuousClock() is not a discrete clock")
end

struct IndexedClock{I}
    clock::TimeDomain
    idx::I
end

Base.getindex(c::TimeDomain, idx) = IndexedClock(c, idx)

function canonicalize_indexed_clock(ic::IndexedClock, sol::AbstractTimeseriesSolution)
    c = ic.clock

    if c isa PeriodicClock
        dt = c.dt
        return ceil(sol.prob.tspan[1] / dt) * dt .+ (ic.idx .- 1) .* dt
    elseif c isa SolverStepClock
        ssc_idx = findfirst(eachindex(sol.discretes)) do i
            !isa(sol.discretes[i].t, AbstractRange)
        end
        return sol.discretes[ssc_idx].t[ic.idx]
    elseif c isa ContinuousClock
        return sol.t[ic.idx]
    else
        error("Unknown clock type: $(typeof(c))")
    end
end

# Define stub functions that can be overridden by extensions
isclock_moshi(c::TimeDomain) = isclock(c)
issolverstepclock_moshi(c::TimeDomain) = issolverstepclock(c)
iscontinuous_moshi(c::TimeDomain) = iscontinuous(c)
first_clock_tick_time_moshi(c, t0) = first_clock_tick_time(c, t0)
function canonicalize_indexed_clock_moshi(ic::IndexedClock, sol::AbstractTimeseriesSolution)
    canonicalize_indexed_clock(ic, sol)
end
