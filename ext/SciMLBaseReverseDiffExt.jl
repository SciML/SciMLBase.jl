module SciMLBaseReverseDiffExt

function DiffEqBase.anyeltypedual(::Type{T},
        ::Type{Val{counter}} = Val{0}) where {counter} where {
        V, D, N, VA, DA, T <: ReverseDiff.TrackedArray{V, D, N, VA, DA}}
    DiffEqBase.anyeltypedual(V, Val{counter})
end

DiffEqBase.value(x::Type{ReverseDiff.TrackedReal{V, D, O}}) where {V, D, O} = V
function DiffEqBase.value(x::Type{
        ReverseDiff.TrackedArray{V, D, N, VA, DA},
}) where {V, D,
        N, VA,
        DA}
    Array{V, N}
end
DiffEqBase.value(x::ReverseDiff.TrackedReal) = x.value
DiffEqBase.value(x::ReverseDiff.TrackedArray) = x.value

DiffEqBase.unitfulvalue(x::Type{ReverseDiff.TrackedReal{V, D, O}}) where {V, D, O} = V
function DiffEqBase.unitfulvalue(x::Type{
        ReverseDiff.TrackedArray{V, D, N, VA, DA},
}) where {V, D,
        N, VA,
        DA}
    Array{V, N}
end
DiffEqBase.unitfulvalue(x::ReverseDiff.TrackedReal) = x.value
DiffEqBase.unitfulvalue(x::ReverseDiff.TrackedArray) = x.value

# Force TrackedArray from TrackedReal when reshaping W\b
DiffEqBase._reshape(v::AbstractVector{<:ReverseDiff.TrackedReal}, siz) = reduce(vcat, v)

DiffEqBase.promote_u0(u0::ReverseDiff.TrackedArray, p::ReverseDiff.TrackedArray, t0) = u0
function DiffEqBase.promote_u0(u0::AbstractArray{<:ReverseDiff.TrackedReal},
        p::ReverseDiff.TrackedArray, t0)
    u0
end
function DiffEqBase.promote_u0(u0::ReverseDiff.TrackedArray,
        p::AbstractArray{<:ReverseDiff.TrackedReal}, t0)
    u0
end
function DiffEqBase.promote_u0(u0::AbstractArray{<:ReverseDiff.TrackedReal},
        p::AbstractArray{<:ReverseDiff.TrackedReal}, t0)
    u0
end
DiffEqBase.promote_u0(u0, p::ReverseDiff.TrackedArray, t0) = ReverseDiff.track(u0)
function DiffEqBase.promote_u0(
        u0, p::ReverseDiff.TrackedArray{T}, t0) where {T <: ReverseDiff.ForwardDiff.Dual}
    ReverseDiff.track(T.(u0))
end
DiffEqBase.promote_u0(u0, p::AbstractArray{<:ReverseDiff.TrackedReal}, t0) = eltype(p).(u0)

end