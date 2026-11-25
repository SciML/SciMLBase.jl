"""
$(TYPEDEF)

Problem wrapper for sampling or replaying an `AbstractNoiseProcess`.

`NoiseProblem` stores the noise process, the time span over which it should be
generated, a deterministic seed, and solver keyword arguments. Noise-process
solvers use this problem type to construct a saved noise trajectory that follows
the common SciML solution interface.

# Fields

$(TYPEDFIELDS)
"""
struct NoiseProblem{N <: AbstractNoiseProcess, T, K} <: AbstractNoiseProblem
    noise::N
    tspan::T
    seed::UInt64
    kwargs::K
end

@add_kwonly function NoiseProblem(noise, tspan; seed = UInt64(0), kwargs...)
    _tspan = promote_tspan(tspan)
    NoiseProblem{typeof(noise), typeof(_tspan), typeof(kwargs)}(noise, _tspan, seed, kwargs)
end
