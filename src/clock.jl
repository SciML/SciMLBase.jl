module Clocks

export TimeDomain

using Expronicon.ADT: variant_type, @adt, @match

@adt TimeDomain begin
    Continuous
    struct PeriodicClock
        dt::Union{Nothing, Float64, Rational{Int}}
        phase::Float64 = 0.0
    end
    SolverStepClock
end

Base.Broadcast.broadcastable(d::TimeDomain) = Ref(d)

const DiscriminatorType = typeof(variant_type(Continuous))

function Base.write(io::IO, x::DiscriminatorType)
    write(io, Base.reinterpret(UInt32, x))
end

function Base.read(io::IO, ::Type{DiscriminatorType})
    Base.reinterpret(DiscriminatorType, read(io, UInt32))
end

end

using .Clocks

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
    PeriodicClock(_...) => true
    _ => false
end

issolverstepclock(c) = @match c begin
    &SolverStepClock => true
    _ => false
end

iscontinuous(c) = @match c begin
    &Continuous => true
    _ => false
end

is_discrete_time_domain(c) = !iscontinuous(c)

function first_clock_tick_time(c, t0)
    @match c begin
        PeriodicClock(dt, _...) => ceil(t0 / dt) * dt
        &SolverStepClock => t0
        &Continuous => error("Continuous is not a discrete clock")
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
        PeriodicClock(dt, _...) => ceil(sol.prob.tspan[1] / dt) * dt .+ (ic.idx .- 1) .* dt
        &SolverStepClock => begin
            ssc_idx = findfirst(eachindex(sol.discretes)) do i
                !isa(sol.discretes[i].t, AbstractRange)
            end
            sol.discretes[ssc_idx].t[ic.idx]
        end
        &Continuous => sol.t[ic.idx]
    end
end
