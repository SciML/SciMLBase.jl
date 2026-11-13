module SciMLBaseMeasurementsExt

using Measurements: Measurements
using SciMLBase: SciMLBase

function SciMLBase.promote_u0(
        u0::AbstractArray{<:Measurements.Measurement},
        p::AbstractArray{<:Measurements.Measurement}, t0
    )
    return u0
end
SciMLBase.promote_u0(u0, p::AbstractArray{<:Measurements.Measurement}, t0) = eltype(p).(u0)

SciMLBase.value(x::Type{Measurements.Measurement{T}}) where {T} = SciMLBase.value(T)
SciMLBase.value(x::Measurements.Measurement) = SciMLBase.value(Measurements.value(x))

function SciMLBase.unitfulvalue(x::Type{Measurements.Measurement{T}}) where {T}
    return SciMLBase.unitfulvalue(T)
end
function SciMLBase.unitfulvalue(x::Measurements.Measurement)
    return SciMLBase.unitfulvalue(Measurements.value(x))
end

end
