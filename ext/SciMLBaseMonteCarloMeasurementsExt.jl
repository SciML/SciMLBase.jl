module SciMLBaseMonteCarloMeasurementsExt

using MonteCarloMeasurements: MonteCarloMeasurements
using SciMLBase: SciMLBase
using Statistics: mean

function SciMLBase.promote_u0(
        u0::AbstractArray{
            <:MonteCarloMeasurements.AbstractParticles,
        },
        p::AbstractArray{<:MonteCarloMeasurements.AbstractParticles},
        t0
    )
    return u0
end
function SciMLBase.promote_u0(
        u0,
        p::AbstractArray{<:MonteCarloMeasurements.AbstractParticles},
        t0
    )
    return eltype(p).(u0)
end

function SciMLBase.promote_u0(
        ::Nothing,
        p::AbstractArray{<:MonteCarloMeasurements.AbstractParticles},
        t0
    )
    return nothing
end

function SciMLBase.value(
        x::Type{<:MonteCarloMeasurements.AbstractParticles{T}}
    ) where {T}
    return SciMLBase.value(T)
end
function SciMLBase.value(x::MonteCarloMeasurements.AbstractParticles)
    return SciMLBase.value(mean(x.particles))
end
function SciMLBase.unitfulvalue(
        x::Type{<:MonteCarloMeasurements.AbstractParticles{T}}
    ) where {T}
    return SciMLBase.unitfulvalue(T)
end
function SciMLBase.unitfulvalue(x::MonteCarloMeasurements.AbstractParticles)
    return SciMLBase.unitfulvalue(mean(x.particles))
end

end
