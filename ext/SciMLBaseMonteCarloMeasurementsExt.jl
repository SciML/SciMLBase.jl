module SciMLBaseMonteCarloMeasurementsExt

using SciMLBase
using SciMLBase: value
using MonteCarloMeasurements

function SciMLBase.promote_u0(
        u0::AbstractArray{
            <:MonteCarloMeasurements.AbstractParticles,
        },
        p::AbstractArray{<:MonteCarloMeasurements.AbstractParticles},
        t0)
    u0
end
function SciMLBase.promote_u0(u0,
        p::AbstractArray{<:MonteCarloMeasurements.AbstractParticles},
        t0)
    eltype(p).(u0)
end

function SciMLBase.promote_u0(::Nothing,
        p::AbstractArray{<:MonteCarloMeasurements.AbstractParticles},
        t0)
    return nothing
end

SciMLBase.value(x::Type{MonteCarloMeasurements.AbstractParticles{T, N}}) where {T, N} = T
SciMLBase.value(x::MonteCarloMeasurements.AbstractParticles) = mean(x.particles)
function SciMLBase.unitfulvalue(x::Type{MonteCarloMeasurements.AbstractParticles{
        T, N}}) where {T, N}
    T
end
SciMLBase.unitfulvalue(x::MonteCarloMeasurements.AbstractParticles) = mean(x.particles)

end
