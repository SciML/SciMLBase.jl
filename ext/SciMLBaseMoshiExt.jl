module SciMLBaseMoshiExt

using SciMLBase
using Moshi.Data: @data
using Moshi.Match: @match

# Define the original Clocks ADT using Moshi - this recreates the exact original implementation
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

# Define constructors
function SciMLBase.Clock(dt::Union{<:Rational, Float64}; phase = 0.0)
    PeriodicClock(dt, phase)
end

function SciMLBase.Clock(dt; phase = 0.0)
    PeriodicClock(convert(Float64, dt), phase)
end

function SciMLBase.Clock(; phase = 0.0)
    PeriodicClock(nothing, phase)
end

# Define clock type checking functions using pattern matching
function SciMLBase.isclock(c::TimeDomain)
    @match c begin
        PeriodicClock() => true
        _ => false
    end
end

function SciMLBase.issolverstepclock(c::TimeDomain)
    @match c begin
        SolverStepClock() => true
        _ => false
    end
end

function SciMLBase.iscontinuous(c::TimeDomain)
    @match c begin
        ContinuousClock() => true
        _ => false
    end
end

SciMLBase.is_discrete_time_domain(c::TimeDomain) = !SciMLBase.iscontinuous(c)

function SciMLBase.first_clock_tick_time(c::TimeDomain, t0)
    @match c begin
        PeriodicClock(dt) => ceil(t0 / dt) * dt
        SolverStepClock() => t0
        ContinuousClock() => error("ContinuousClock() is not a discrete clock")
    end
end

# Additional functionality
Base.Broadcast.broadcastable(d::TimeDomain) = Ref(d)
(clock::TimeDomain)() = clock

struct IndexedClock{I}
    clock::TimeDomain
    idx::I
end

Base.getindex(c::TimeDomain, idx) = IndexedClock(c, idx)

function SciMLBase.canonicalize_indexed_clock(ic::IndexedClock, sol::SciMLBase.AbstractTimeseriesSolution)
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

end