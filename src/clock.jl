@data Clocks begin
    ContinuousClock
    struct PeriodicClock
        dt::Union{Nothing, Float64, Rational{Int}}
        phase::Float64 = 0.0
    end
    SolverStepClock
end

# for backwards compatibility
const TimeDomain = Clocks.Type
using .Clocks: ContinuousClock, PeriodicClock, SolverStepClock
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

isclock(c) = @match c begin
    PeriodicClock() => true
    _ => false
end

issolverstepclock(c) = @match c begin
    SolverStepClock() => true
    _ => false
end

iscontinuous(c) = @match c begin
    ContinuousClock() => true
    _ => false
end

is_discrete_time_domain(c) = !iscontinuous(c)

function first_clock_tick_time(c, t0)
    @match c begin
        PeriodicClock(dt) => ceil(t0 / dt) * dt
        SolverStepClock() => t0
        ContinuousClock() => error("ContinuousClock() is not a discrete clock")
    end
end

struct IndexedClock{I}
    clock::TimeDomain
    idx::I
end

Base.getindex(c::TimeDomain, idx) = IndexedClock(c, idx)

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
