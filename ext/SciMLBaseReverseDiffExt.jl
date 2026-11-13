module SciMLBaseReverseDiffExt

using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using SciMLBase: SciMLBase

function SciMLBase.anyeltypedual(
        ::Type{T},
        ::Type{Val{counter}} = Val{0}
    ) where {counter} where {
        V, D, N, VA, DA, T <: ReverseDiff.TrackedArray{V, D, N, VA, DA},
    }
    return SciMLBase.anyeltypedual(V, Val{counter})
end

function SciMLBase.value(x::Type{ReverseDiff.TrackedReal{V, D, O}}) where {V, D, O}
    return SciMLBase.value(V)
end
function SciMLBase.value(
        x::Type{
            ReverseDiff.TrackedArray{V, D, N, VA, DA},
        }
    ) where {
        V, D,
        N, VA,
        DA,
    }
    return SciMLBase.value(Array{V, N})
end
SciMLBase.value(x::ReverseDiff.TrackedReal) = SciMLBase.value(x.value)
SciMLBase.value(x::ReverseDiff.TrackedArray) = SciMLBase.value(x.value)

function SciMLBase.unitfulvalue(x::Type{ReverseDiff.TrackedReal{V, D, O}}) where {V, D, O}
    return SciMLBase.unitfulvalue(V)
end
function SciMLBase.unitfulvalue(
        x::Type{
            ReverseDiff.TrackedArray{V, D, N, VA, DA},
        }
    ) where {
        V, D,
        N, VA,
        DA,
    }
    return SciMLBase.unitfulvalue(Array{V, N})
end
SciMLBase.unitfulvalue(x::ReverseDiff.TrackedReal) = SciMLBase.unitfulvalue(x.value)
SciMLBase.unitfulvalue(x::ReverseDiff.TrackedArray) = SciMLBase.unitfulvalue(x.value)

# Force TrackedArray from TrackedReal when reshaping W\b
SciMLBase._reshape(v::AbstractVector{<:ReverseDiff.TrackedReal}, siz) = reduce(vcat, v)

SciMLBase.promote_u0(u0::ReverseDiff.TrackedArray, p::ReverseDiff.TrackedArray, t0) = u0
function SciMLBase.promote_u0(
        u0::AbstractArray{<:ReverseDiff.TrackedReal},
        p::ReverseDiff.TrackedArray, t0
    )
    return u0
end
function SciMLBase.promote_u0(
        u0::ReverseDiff.TrackedArray,
        p::AbstractArray{<:ReverseDiff.TrackedReal}, t0
    )
    return u0
end
function SciMLBase.promote_u0(
        u0::AbstractArray{<:ReverseDiff.TrackedReal},
        p::AbstractArray{<:ReverseDiff.TrackedReal}, t0
    )
    return u0
end
SciMLBase.promote_u0(u0, p::ReverseDiff.TrackedArray, t0) = ReverseDiff.track(u0)
function SciMLBase.promote_u0(
        u0, p::ReverseDiff.TrackedArray{T}, t0
    ) where {T <: ForwardDiff.Dual}
    return ReverseDiff.track(T.(u0))
end
SciMLBase.promote_u0(u0, p::AbstractArray{<:ReverseDiff.TrackedReal}, t0) = eltype(p).(u0)

end
