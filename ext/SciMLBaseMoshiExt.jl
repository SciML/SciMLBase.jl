module SciMLBaseMoshiExt

using SciMLBase
using Moshi.Data: @data
using Moshi.Match: @match

# This extension replaces the fallback Clocks module with the original Moshi-based implementation
# This gives pattern matching capability when Moshi is loaded
@data Clocks begin
    ContinuousClock
    struct PeriodicClock
        dt::Union{Nothing, Float64, Rational{Int}}
        phase::Float64 = 0.0
    end
    SolverStepClock
end

# Override SciMLBase.Clocks with the Moshi version when this extension loads
Core.eval(SciMLBase, :(const Clocks = $Clocks))

# for backwards compatibility - these need to be redefined with the new types
Core.eval(SciMLBase, :(const TimeDomain = Clocks.Type))
Core.eval(SciMLBase, :(using .Clocks: ContinuousClock, PeriodicClock, SolverStepClock))
Core.eval(SciMLBase, :(const Continuous = ContinuousClock()))

# Redefine the pattern matching functions to use @match
function SciMLBase.isclock(c::SciMLBase.TimeDomain)
    @match c begin
        PeriodicClock() => true
        _ => false
    end
end

function SciMLBase.issolverstepclock(c::SciMLBase.TimeDomain)
    @match c begin
        SolverStepClock() => true
        _ => false
    end
end

function SciMLBase.iscontinuous(c::SciMLBase.TimeDomain)
    @match c begin
        ContinuousClock() => true
        _ => false
    end
end

function SciMLBase.first_clock_tick_time(c, t0)
    @match c begin
        PeriodicClock(dt) => ceil(t0 / dt) * dt
        SolverStepClock() => t0
        ContinuousClock() => error("ContinuousClock() is not a discrete clock")
        _ => error("Unknown clock type: $(typeof(c))")
    end
end

function SciMLBase.canonicalize_indexed_clock(ic::SciMLBase.IndexedClock, sol::SciMLBase.AbstractTimeseriesSolution)
    c = ic.clock
    
    @match c begin
        PeriodicClock(dt) => ceil(sol.prob.tspan[1] / dt) * dt .+ (ic.idx .- 1) .* dt
        SolverStepClock() => begin
            ssc_idx = findfirst(eachindex(sol.discretes)) do i
                !isa(sol.discretes[i].t, AbstractRange)
            end
            sol.discretes[ssc_idx].t[ic.idx]
        end
        ContinuousClock() => sol.t[ic.idx]
        _ => error("Unknown clock type: $(typeof(c))")
    end
end

end