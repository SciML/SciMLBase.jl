module SciMLBaseMeasurementsExt

using Measurements
using SciMLBase: SciMLBase

function SciMLBase.promote_u0(u0::AbstractArray{<:Measurements.Measurement},
        p::AbstractArray{<:Measurements.Measurement}, t0)
    u0
end
SciMLBase.promote_u0(u0, p::AbstractArray{<:Measurements.Measurement}, t0) = eltype(p).(u0)

SciMLBase.value(x::Type{Measurements.Measurement{T}}) where {T} = T
SciMLBase.value(x::Measurements.Measurement) = Measurements.value(x)

SciMLBase.unitfulvalue(x::Type{Measurements.Measurement{T}}) where {T} = T
SciMLBase.unitfulvalue(x::Measurements.Measurement) = Measurements.value(x)

end
