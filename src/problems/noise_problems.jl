"""
$(TYPEDEF)
"""
struct NoiseProblem{N <: AbstractNoiseProcess, T, K} <: AbstractNoiseProblem
    noise::N
    tspan::T
    seed::UInt64
    kwargs::K
end

function Base.show(io::IO, t::NoiseProblem{N, T, K}) where {N, T, K}
    if TruncatedStacktraces.VERBOSE[]
        print(io, "NoiseProblem{$N,$T,$K}")
    else
        print(io, "NoiseProblem{$N,…}")
    end
end

@add_kwonly function NoiseProblem(noise, tspan; seed = UInt64(0), kwargs...)
    _tspan = promote_tspan(tspan)
    NoiseProblem{typeof(noise), typeof(_tspan), typeof(kwargs)}(noise, _tspan, seed, kwargs)
end
