module SciMLBaseTrackerExt

using SciMLBase: SciMLBase
import Tracker

SciMLBase.value(x::Type{Tracker.TrackedReal{T}}) where {T} = SciMLBase.value(T)
function SciMLBase.value(x::Type{Tracker.TrackedArray{T, N, A}}) where {T, N, A}
    return SciMLBase.value(Array{T, N})
end
SciMLBase.value(x::Tracker.TrackedReal) = SciMLBase.value(x.data)
SciMLBase.value(x::Tracker.TrackedArray) = SciMLBase.value(x.data)

SciMLBase.unitfulvalue(x::Type{Tracker.TrackedReal{T}}) where {T} = SciMLBase.unitfulvalue(T)
function SciMLBase.unitfulvalue(x::Type{Tracker.TrackedArray{T, N, A}}) where {T, N, A}
    return SciMLBase.unitfulvalue(Array{T, N})
end
SciMLBase.unitfulvalue(x::Tracker.TrackedReal) = SciMLBase.unitfulvalue(x.data)
SciMLBase.unitfulvalue(x::Tracker.TrackedArray) = SciMLBase.unitfulvalue(x.data)

SciMLBase.promote_u0(u0::Tracker.TrackedArray, p::Tracker.TrackedArray, t0) = u0
function SciMLBase.promote_u0(
        u0::AbstractArray{<:Tracker.TrackedReal},
        p::Tracker.TrackedArray, t0
    )
    return u0
end
function SciMLBase.promote_u0(
        u0::Tracker.TrackedArray,
        p::AbstractArray{<:Tracker.TrackedReal}, t0
    )
    return u0
end
function SciMLBase.promote_u0(
        u0::AbstractArray{<:Tracker.TrackedReal},
        p::AbstractArray{<:Tracker.TrackedReal}, t0
    )
    return u0
end
SciMLBase.promote_u0(u0, p::Tracker.TrackedArray, t0) = Tracker.track(u0)
SciMLBase.promote_u0(u0, p::AbstractArray{<:Tracker.TrackedReal}, t0) = eltype(p).(u0)

@inline Base.any(f::Function, x::Tracker.TrackedArray) = any(f, Tracker.data(x))


end
