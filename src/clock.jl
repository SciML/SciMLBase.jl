# Basic clock types - fallback implementation when Moshi is not available
abstract type TimeDomain end

struct ContinuousClock <: TimeDomain end
struct SolverStepClock <: TimeDomain end

struct PeriodicClock <: TimeDomain
    dt::Union{Nothing, Float64, Rational{Int}}
    phase::Float64
    
    function PeriodicClock(dt::Union{Nothing, Float64, Rational{Int}}, phase::Float64 = 0.0)
        new(dt, phase)
    end
end

# Construct with keywords
PeriodicClock(; dt = nothing, phase = 0.0) = PeriodicClock(dt, phase)

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

isclock(c::TimeDomain) = c isa PeriodicClock

issolverstepclock(c::TimeDomain) = c isa SolverStepClock

iscontinuous(c::TimeDomain) = c isa ContinuousClock

is_discrete_time_domain(c::TimeDomain) = !iscontinuous(c)

# workaround for https://github.com/Roger-luo/Moshi.jl/issues/43
isclock(::Any) = false
issolverstepclock(::Any) = false
iscontinuous(::Any) = false
is_discrete_time_domain(::Any) = false

function first_clock_tick_time(c, t0)
    if c isa PeriodicClock
        dt = c.dt
        return ceil(t0 / dt) * dt
    elseif c isa SolverStepClock
        return t0
    elseif c isa ContinuousClock
        error("ContinuousClock() is not a discrete clock")
    else
        error("Unknown clock type: $(typeof(c))")
    end
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
