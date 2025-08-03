module SciMLBaseMoshiExt

if isdefined(Base, :get_extension)
    using Moshi.Data: @data
    using Moshi.Match: @match
else
    using ..Moshi.Data: @data
    using ..Moshi.Match: @match
end

using SciMLBase
using SciMLBase: AbstractTimeseriesSolution, TimeDomain, PeriodicClock, SolverStepClock,
                 ContinuousClock

# Enhanced clock predicates using @match when Moshi is available
# These override the fallback implementations with pattern matching
function SciMLBase.isclock_moshi(c::TimeDomain)
    @match c begin
        PeriodicClock() => true
        _ => false
    end
end

function SciMLBase.issolverstepclock_moshi(c::TimeDomain)
    @match c begin
        SolverStepClock() => true
        _ => false
    end
end

function SciMLBase.iscontinuous_moshi(c::TimeDomain)
    @match c begin
        ContinuousClock() => true
        _ => false
    end
end

function SciMLBase.first_clock_tick_time_moshi(c, t0)
    @match c begin
        PeriodicClock(dt) => ceil(t0 / dt) * dt
        SolverStepClock() => t0
        ContinuousClock() => error("ContinuousClock() is not a discrete clock")
    end
end

function SciMLBase.canonicalize_indexed_clock_moshi(ic::SciMLBase.IndexedClock, sol::AbstractTimeseriesSolution)
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

# Override fallback implementations to use Moshi versions
function SciMLBase.isclock(c::TimeDomain)
    SciMLBase.isclock_moshi(c)
end

function SciMLBase.issolverstepclock(c::TimeDomain)
    SciMLBase.issolverstepclock_moshi(c)
end

function SciMLBase.iscontinuous(c::TimeDomain)
    SciMLBase.iscontinuous_moshi(c)
end

function SciMLBase.first_clock_tick_time(c, t0)
    SciMLBase.first_clock_tick_time_moshi(c, t0)
end

function SciMLBase.canonicalize_indexed_clock(ic::SciMLBase.IndexedClock, sol::AbstractTimeseriesSolution)
    SciMLBase.canonicalize_indexed_clock_moshi(ic, sol)
end

end # module
