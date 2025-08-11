module SciMLBaseTrackerExt

using SciMLBase
import Tracker

DiffEqBase.value(x::Type{Tracker.TrackedReal{T}}) where {T} = T
DiffEqBase.value(x::Type{Tracker.TrackedArray{T, N, A}}) where {T, N, A} = Array{T, N}
DiffEqBase.value(x::Tracker.TrackedReal) = x.data
DiffEqBase.value(x::Tracker.TrackedArray) = x.data

DiffEqBase.unitfulvalue(x::Type{Tracker.TrackedReal{T}}) where {T} = T
function DiffEqBase.unitfulvalue(x::Type{Tracker.TrackedArray{T, N, A}}) where {T, N, A}
    Array{T, N}
end
DiffEqBase.unitfulvalue(x::Tracker.TrackedReal) = x.data
DiffEqBase.unitfulvalue(x::Tracker.TrackedArray) = x.data

DiffEqBase.promote_u0(u0::Tracker.TrackedArray, p::Tracker.TrackedArray, t0) = u0
function DiffEqBase.promote_u0(u0::AbstractArray{<:Tracker.TrackedReal},
        p::Tracker.TrackedArray, t0)
    u0
end
function DiffEqBase.promote_u0(u0::Tracker.TrackedArray,
        p::AbstractArray{<:Tracker.TrackedReal}, t0)
    u0
end
function DiffEqBase.promote_u0(u0::AbstractArray{<:Tracker.TrackedReal},
        p::AbstractArray{<:Tracker.TrackedReal}, t0)
    u0
end
DiffEqBase.promote_u0(u0, p::Tracker.TrackedArray, t0) = Tracker.track(u0)
DiffEqBase.promote_u0(u0, p::AbstractArray{<:Tracker.TrackedReal}, t0) = eltype(p).(u0)

@inline Base.any(f::Function, x::Tracker.TrackedArray) = any(f, Tracker.data(x))


end