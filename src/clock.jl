module Clocks

export TimeDomain

using Expronicon.ADT: @adt, @match

@adt TimeDomain begin
    Continuous
    struct PeriodicClock
        t
        dt::Union{Nothing, Float64}
    end
    struct SolverStepClock
        t = nothing
    end
end

Base.hash(c::TimeDomain, seed::UInt) = @match c begin
    PeriodicClock(_, dt) => hash(dt, seed ⊻ 0x953d7a9a18874b90)
    SolverStepClock(_) => seed ⊻ 0x953d7a9a18874b90
    &Continuous => seed ⊻ 0xfd2a3dfeb13318e5
end

function Base.:(==)(c1::TimeDomain, c2::TimeDomain)
    @match c1 begin
        PeriodicClock(t1, dt1) => @match c2 begin
            PeriodicClock(t2, dt2) => (t1 === nothing || t2 === nothing || isequal(t1, t2)) && dt1 == dt2
            _ => false
        end
        SolverStepClock(t1) => @match c2 begin
            SolverStepClock(t2) => t1 === nothing || t2 === nothing || isequal(t1, t2)
            _ => false
        end
        &Continuous => iscontinuous(c2)
    end
end

Base.Broadcast.broadcastable(d::TimeDomain) = Ref(d)

end

using .Clocks

"""
    Clock(t, dt)
    Clock(t)
    Clock(dt)
    Clock()

The default periodic clock with symbolic independent variable `t` and tick interval `dt`.
If `dt` is left unspecified, it will be inferred (if possible). If `t` is not specified,
it is `nothing`.
"""
Clock(t, dt) = PeriodicClock(t, dt)
function Clock(arg)
    if symbolic_type(arg) == NotSymbolic()
        return PeriodicClock(nothing, arg)
    else
        return PeriodicClock(arg, nothing)
    end
end
Clock() = PeriodicClock(nothing, nothing)

@doc """
    SolverStepClock(t)
    SolverStepClock()

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
    SolverStepClock(_...) => true
    _ => false
end

iscontinuous(c) = @match c begin
    &Continuous => true
    _ => false
end

is_discrete_time_domain(c) = !iscontinuous(c)

struct IndexedClock{I}
    clock::TimeDomain
    idx::I
end

Base.getindex(c::TimeDomain, idx) = IndexedClock(c, idx)

function canonicalize_indexed_clock(ic::IndexedClock, sol::AbstractTimeseriesSolution)
    c = ic.clock
    
    return @match c begin
        PeriodicClock(_, dt) => ceil(sol.prob.tspan[1] / dt) * dt .+ (ic.idx .- 1) .* dt
        SolverStepClock(_) => begin
            ssc_idx = findfirst(eachindex(sol.discretes)) do i
                !isa(sol.discretes[i].t, AbstractRange)
            end
            sol.discretes[ssc_idx].t[ic.idx]
        end
        &Continuous => sol.t[ic.idx]
    end
end
