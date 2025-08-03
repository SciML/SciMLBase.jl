# Clock functionality is provided by the Moshi extension
# This file provides placeholder definitions that error when Moshi is not available

# Error message constant
const MOSHI_ERROR = "Clock functionality requires Moshi.jl. Please run `using Moshi` or `import Moshi` to enable clock features."

# Placeholder types for exports - these will be replaced by the extension
abstract type TimeDomain end
module Clocks end

# Placeholder constants
const Continuous = nothing
const ContinuousClock = nothing  
const PeriodicClock = nothing
const SolverStepClock = nothing
const IndexedClock = nothing

function Clock(args...; kwargs...)
    error(MOSHI_ERROR)
end

function isclock(::Any)
    error(MOSHI_ERROR)
end

function issolverstepclock(::Any)
    error(MOSHI_ERROR)
end

function iscontinuous(::Any)
    error(MOSHI_ERROR)
end

function is_discrete_time_domain(::Any)
    error(MOSHI_ERROR)
end

function first_clock_tick_time(::Any, ::Any)
    error(MOSHI_ERROR)
end

function canonicalize_indexed_clock(::Any, ::Any)
    error(MOSHI_ERROR)
end
