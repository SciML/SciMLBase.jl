module SciMLBaseMoshiExt

using SciMLBase
using Moshi.Data: @data
using Moshi.Match: @match

# When Moshi is available, override the basic implementations with @match-based versions
# This provides the enhanced pattern matching functionality

function SciMLBase.isclock(c::SciMLBase.TimeDomain)
    @match c begin
        SciMLBase.PeriodicClock() => true
        _ => false
    end
end

function SciMLBase.issolverstepclock(c::SciMLBase.TimeDomain)
    @match c begin
        SciMLBase.SolverStepClock() => true
        _ => false
    end
end

function SciMLBase.iscontinuous(c::SciMLBase.TimeDomain)
    @match c begin
        SciMLBase.ContinuousClock() => true
        _ => false
    end
end

function SciMLBase.first_clock_tick_time(c::SciMLBase.TimeDomain, t0)
    @match c begin
        SciMLBase.PeriodicClock(dt) => ceil(t0 / dt) * dt
        SciMLBase.SolverStepClock() => t0
        SciMLBase.ContinuousClock() => error("ContinuousClock() is not a discrete clock")
    end
end

function SciMLBase.canonicalize_indexed_clock(ic::SciMLBase.IndexedClock, sol::SciMLBase.AbstractTimeseriesSolution)
    c = ic.clock

    return @match c begin
        SciMLBase.PeriodicClock(dt) => ceil(sol.prob.tspan[1] / dt) * dt .+ (ic.idx .- 1) .* dt
        SciMLBase.SolverStepClock() => begin
            ssc_idx = findfirst(eachindex(sol.discretes)) do i
                !isa(sol.discretes[i].t, AbstractRange)
            end
            sol.discretes[ssc_idx].t[ic.idx]
        end
        SciMLBase.ContinuousClock() => sol.t[ic.idx]
    end
end

# Also define Moshi-based types for users who want the original @data experience
@data MoshiClocks begin
    MoshiContinuousClock
    struct MoshiPeriodicClock
        dt::Union{Nothing, Float64, Rational{Int}}
        phase::Float64 = 0.0
    end
    MoshiSolverStepClock
end

# Convenience constructors for the Moshi types
MoshiClock(dt::Union{<:Rational, Float64}; phase = 0.0) = MoshiPeriodicClock(dt, phase)
MoshiClock(dt; phase = 0.0) = MoshiPeriodicClock(convert(Float64, dt), phase)
MoshiClock(; phase = 0.0) = MoshiPeriodicClock(nothing, phase)

end