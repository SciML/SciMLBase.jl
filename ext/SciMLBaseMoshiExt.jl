module SciMLBaseMoshiExt

using SciMLBase
using Moshi.Data: @data
using Moshi.Match: @match

# Define Moshi-based clock types that provide enhanced pattern matching
@data MoshiClocks begin
    MoshiContinuousClock
    struct MoshiPeriodicClock
        dt::Union{Nothing, Float64, Rational{Int}}
        phase::Float64 = 0.0
    end
    MoshiSolverStepClock
end

# Define type alias for Moshi clock types
const MoshiTimeDomain = MoshiClocks.Type
using .MoshiClocks: MoshiContinuousClock, MoshiPeriodicClock, MoshiSolverStepClock

# Add pattern matching methods for the Moshi types
SciMLBase.isclock(c::MoshiTimeDomain) = @match c begin
    MoshiPeriodicClock() => true
    _ => false
end

SciMLBase.issolverstepclock(c::MoshiTimeDomain) = @match c begin
    MoshiSolverStepClock() => true
    _ => false
end

SciMLBase.iscontinuous(c::MoshiTimeDomain) = @match c begin
    MoshiContinuousClock() => true
    _ => false
end

function SciMLBase.first_clock_tick_time(c::MoshiTimeDomain, t0)
    @match c begin
        MoshiPeriodicClock(dt) => ceil(t0 / dt) * dt
        MoshiSolverStepClock() => t0
        MoshiContinuousClock() => error("ContinuousClock() is not a discrete clock")
    end
end

function SciMLBase.canonicalize_indexed_clock(ic::SciMLBase.IndexedClock, sol::SciMLBase.AbstractTimeseriesSolution)
    c = ic.clock
    
    if c isa MoshiTimeDomain
        @match c begin
            MoshiPeriodicClock(dt) => ceil(sol.prob.tspan[1] / dt) * dt .+ (ic.idx .- 1) .* dt
            MoshiSolverStepClock() => begin
                ssc_idx = findfirst(eachindex(sol.discretes)) do i
                    !isa(sol.discretes[i].t, AbstractRange)
                end
                sol.discretes[ssc_idx].t[ic.idx]
            end
            MoshiContinuousClock() => sol.t[ic.idx]
        end
    else
        # Fallback to default implementation for non-Moshi types
        invoke(SciMLBase.canonicalize_indexed_clock, Tuple{SciMLBase.IndexedClock, SciMLBase.AbstractTimeseriesSolution}, ic, sol)
    end
end

# Convenience constructors for users who want to explicitly use Moshi types
MoshiClock(dt::Union{<:Rational, Float64}; phase = 0.0) = MoshiPeriodicClock(dt, phase)
MoshiClock(dt; phase = 0.0) = MoshiPeriodicClock(convert(Float64, dt), phase)
MoshiClock(; phase = 0.0) = MoshiPeriodicClock(nothing, phase)

# Export the Moshi types for users who want them
const MoshiContinuous = MoshiContinuousClock()

end