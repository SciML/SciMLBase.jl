# Clock functionality requires Moshi extension - original implementation was:
#
# @data Clocks begin
#     ContinuousClock
#     struct PeriodicClock
#         dt::Union{Nothing, Float64, Rational{Int}}
#         phase::Float64 = 0.0
#     end
#     SolverStepClock
# end
# 
# This is recreated by the SciMLBaseMoshiExt when Moshi is loaded

# Create a module with the same structure as the original, but with stub implementations
module Clocks
    abstract type Type end
    
    struct ContinuousClock <: Type end
    struct SolverStepClock <: Type end
    
    struct PeriodicClock <: Type
        dt::Union{Nothing, Float64, Rational{Int}}
        phase::Float64
        PeriodicClock(dt, phase = 0.0) = new(dt, phase)
    end
    
    # Keyword constructor
    PeriodicClock(; dt = nothing, phase = 0.0) = PeriodicClock(dt, phase)
end

# Re-export for backwards compatibility  
const TimeDomain = Clocks.Type
using .Clocks: ContinuousClock, PeriodicClock, SolverStepClock
const Continuous = ContinuousClock()

# These will be overridden by the extension with @match-based versions
Clock(dt::Union{<:Rational, Float64}; phase = 0.0) = PeriodicClock(dt, phase)
Clock(dt; phase = 0.0) = PeriodicClock(convert(Float64, dt), phase)
Clock(; phase = 0.0) = PeriodicClock(nothing, phase)

isclock(c::TimeDomain) = c isa PeriodicClock
issolverstepclock(c::TimeDomain) = c isa SolverStepClock
iscontinuous(c::TimeDomain) = c isa ContinuousClock
is_discrete_time_domain(c::TimeDomain) = !iscontinuous(c)

# Fallbacks for non-TimeDomain types
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

Base.Broadcast.broadcastable(d::TimeDomain) = Ref(d)
(clock::TimeDomain)() = clock

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
