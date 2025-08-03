module SciMLBaseMoshiExt

using SciMLBase
using Moshi.Data: @data
using Moshi.Match: @match

# Define Moshi-based clock types that can take advantage of pattern matching
@data MoshiClocks begin
    MoshiContinuousClock
    struct MoshiPeriodicClock
        dt::Union{Nothing, Float64, Rational{Int}}
        phase::Float64 = 0.0
    end
    MoshiSolverStepClock
end

# for backwards compatibility
const MoshiTimeDomain = MoshiClocks.Type
using .MoshiClocks: MoshiContinuousClock, MoshiPeriodicClock, MoshiSolverStepClock

# Create instances that use the Moshi pattern matching
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

function SciMLBase.canonicalize_indexed_clock(ic::SciMLBase.IndexedClock{I}, sol::SciMLBase.AbstractTimeseriesSolution) where I
    c = ic.clock
    
    if c isa MoshiTimeDomain
        return @match c begin
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
        # fallback to basic implementation
        invoke(SciMLBase.canonicalize_indexed_clock, Tuple{SciMLBase.IndexedClock, SciMLBase.AbstractTimeseriesSolution}, ic, sol)
    end
end

# Create convenience constructors that use Moshi types
function SciMLBase.Clock(dt::Union{<:Rational, Float64}; phase = 0.0, use_moshi = true)
    use_moshi ? MoshiPeriodicClock(dt, phase) : SciMLBase.PeriodicClock(dt, phase)
end

function SciMLBase.Clock(dt; phase = 0.0, use_moshi = true)
    use_moshi ? MoshiPeriodicClock(convert(Float64, dt), phase) : SciMLBase.PeriodicClock(convert(Float64, dt), phase)
end

function SciMLBase.Clock(; phase = 0.0, use_moshi = true)
    use_moshi ? MoshiPeriodicClock(nothing, phase) : SciMLBase.PeriodicClock(nothing, phase)
end

# Export types for those who want to explicitly use Moshi types
const ContinuousClockMoshi = MoshiContinuousClock()
const SolverStepClockMoshi = MoshiSolverStepClock()

end