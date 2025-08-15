module SciMLBaseMeasurementsExt

using Measurements
using SciMLBase: value

function SciMLBase.promote_u0(u0::AbstractArray{<:Measurements.Measurement},
        p::AbstractArray{<:Measurements.Measurement}, t0)
    u0
end
SciMLBase.promote_u0(u0, p::AbstractArray{<:Measurements.Measurement}, t0) = eltype(p).(u0)

value(x::Type{Measurements.Measurement{T}}) where {T} = T
value(x::Measurements.Measurement) = Measurements.value(x)

unitfulvalue(x::Type{Measurements.Measurement{T}}) where {T} = T
unitfulvalue(x::Measurements.Measurement) = Measurements.value(x)

end
