"""
$(TYPEDEF)
"""
struct DiffEqIdentity{T, N} <: AbstractDiffEqLinearOperator{T} end

DiffEqIdentity(u) = DiffEqIdentity{eltype(u), length(u)}()
Base.size(::DiffEqIdentity{T, N}) where {T, N} = (N, N)
Base.size(::DiffEqIdentity{T, N}, m::Integer) where {T, N} = (m == 1 || m == 2) ? N : 1
LinearAlgebra.opnorm(::DiffEqIdentity{T, N}, p::Real = 2) where {T, N} = one(T)
function Base.convert(::Type{AbstractMatrix}, ::DiffEqIdentity{T, N}) where {T, N}
    LinearAlgebra.Diagonal(ones(T, N))
end

for op in (:*, :/, :\)
    @eval Base.$op(::DiffEqIdentity{T, N}, x::AbstractVecOrMat) where {T, N} = $op(I, x)
    @eval Base.$op(::DiffEqIdentity{T, N}, x::AbstractArray) where {T, N} = $op(I, x)
    @eval Base.$op(x::AbstractVecOrMat, ::DiffEqIdentity{T, N}) where {T, N} = $op(x, I)
    @eval Base.$op(x::AbstractArray, ::DiffEqIdentity{T, N}) where {T, N} = $op(x, I)
end
LinearAlgebra.mul!(Y::AbstractVecOrMat, ::DiffEqIdentity, B::AbstractVecOrMat) = Y .= B
LinearAlgebra.ldiv!(Y::AbstractVecOrMat, ::DiffEqIdentity, B::AbstractVecOrMat) = Y .= B

LinearAlgebra.mul!(Y::AbstractArray, ::DiffEqIdentity, B::AbstractArray) = Y .= B
LinearAlgebra.ldiv!(Y::AbstractArray, ::DiffEqIdentity, B::AbstractArray) = Y .= B

for pred in (:isreal, :issymmetric, :ishermitian, :isposdef)
    @eval LinearAlgebra.$pred(::DiffEqIdentity) = true
end

"""
    DiffEqScalar(val[; update_func])

Represents a time-dependent scalar/scaling operator. The update function
is called by `update_coefficients!` and is assumed to have the following
signature:

    update_func(oldval,u,p,t) -> newval
"""
mutable struct DiffEqScalar{T <: Number, F} <: AbstractDiffEqLinearOperator{T}
    val::T
    update_func::F
    function DiffEqScalar(val::T; update_func = DEFAULT_UPDATE_FUNC) where {T}
        new{T, typeof(update_func)}(val, update_func)
    end
end

Base.convert(::Type{Number}, α::DiffEqScalar) = α.val
Base.convert(::Type{DiffEqScalar}, α::Number) = DiffEqScalar(α)
Base.size(::DiffEqScalar) = ()
Base.size(::DiffEqScalar, ::Integer) = 1
update_coefficients!(α::DiffEqScalar, u, p, t) = (α.val = α.update_func(α.val, u, p, t);
α)
isconstant(α::DiffEqScalar) = α.update_func == DEFAULT_UPDATE_FUNC

for op in (:*, :/, :\)
    for T in (
        ### added in https://github.com/SciML/SciMLBase.jl/pull/377
        :AbstractVecOrMat,
        ###
        :AbstractArray,
        :Number)
        @eval Base.$op(α::DiffEqScalar, x::$T) = $op(α.val, x)
        @eval Base.$op(x::$T, α::DiffEqScalar) = $op(x, α.val)
    end
    @eval Base.$op(x::DiffEqScalar, y::DiffEqScalar) = $op(x.val, y.val)
end

for op in (:-, :+)
    @eval Base.$op(α::DiffEqScalar, x::Number) = $op(α.val, x)
    @eval Base.$op(x::Number, α::DiffEqScalar) = $op(x, α.val)
    @eval Base.$op(x::DiffEqScalar, y::DiffEqScalar) = $op(x.val, y.val)
end

LinearAlgebra.lmul!(α::DiffEqScalar, B::AbstractArray) = lmul!(α.val, B)
LinearAlgebra.rmul!(B::AbstractArray, α::DiffEqScalar) = rmul!(B, α.val)
LinearAlgebra.mul!(Y::AbstractArray, α::DiffEqScalar, B::AbstractArray) = mul!(Y, α.val, B)
function LinearAlgebra.axpy!(α::DiffEqScalar, X::AbstractArray, Y::AbstractArray)
    axpy!(α.val, X, Y)
end
Base.abs(α::DiffEqScalar) = abs(α.val)

"""
    DiffEqArrayOperator(A[; update_func])

Represents a time-dependent linear operator given by an AbstractMatrix. The
update function is called by `update_coefficients!` and is assumed to have
the following signature:

    update_func(A::AbstractMatrix,u,p,t) -> [modifies A]
"""
struct DiffEqArrayOperator{T, AType <: AbstractMatrix{T}, F} <:
       AbstractDiffEqLinearOperator{T}
    A::AType
    update_func::F
    function DiffEqArrayOperator(A::AType; update_func = DEFAULT_UPDATE_FUNC) where {AType}
        new{eltype(A), AType, typeof(update_func)}(A, update_func)
    end
end

has_adjoint(::DiffEqArrayOperator) = true
update_coefficients!(L::DiffEqArrayOperator, u, p, t) = (L.update_func(L.A, u, p, t); L)
isconstant(L::DiffEqArrayOperator) = L.update_func == DEFAULT_UPDATE_FUNC
function Base.similar(L::DiffEqArrayOperator, ::Type{T}, dims::Dims) where {T}
    similar(L.A, T, dims)
end
function Base.adjoint(L::DiffEqArrayOperator)
    DiffEqArrayOperator(L.A'; update_func = (A, u, p, t) -> L.update_func(L.A, u, p, t)')
end

# propagate_inbounds here for the getindex fallback
Base.@propagate_inbounds Base.convert(::Type{AbstractMatrix}, L::DiffEqArrayOperator) = L.A
Base.@propagate_inbounds Base.setindex!(L::DiffEqArrayOperator, v, i::Int) = (L.A[i] = v)
Base.@propagate_inbounds function Base.setindex!(L::DiffEqArrayOperator, v,
        I::Vararg{Int, N}) where {N}
    (L.A[I...] = v)
end

Base.eachcol(L::DiffEqArrayOperator) = eachcol(L.A)
Base.eachrow(L::DiffEqArrayOperator) = eachrow(L.A)
Base.length(L::DiffEqArrayOperator) = length(L.A)
Base.iterate(L::DiffEqArrayOperator, args...) = iterate(L.A, args...)
Base.axes(L::DiffEqArrayOperator) = axes(L.A)
Base.eachindex(L::DiffEqArrayOperator) = eachindex(L.A)
function Base.IndexStyle(::Type{<:DiffEqArrayOperator{T, AType}}) where {T, AType}
    Base.IndexStyle(AType)
end
Base.copyto!(L::DiffEqArrayOperator, rhs) = (copyto!(L.A, rhs); L)
Base.Broadcast.broadcastable(L::DiffEqArrayOperator) = L
Base.ndims(::Type{<:DiffEqArrayOperator{T, AType}}) where {T, AType} = ndims(AType)
ArrayInterface.issingular(L::DiffEqArrayOperator) = ArrayInterface.issingular(L.A)
function Base.copy(L::DiffEqArrayOperator)
    DiffEqArrayOperator(copy(L.A); update_func = L.update_func)
end

const AdjointFact = isdefined(LinearAlgebra, :AdjointFactorization) ?
                    LinearAlgebra.AdjointFactorization : Adjoint
const TransposeFact = isdefined(LinearAlgebra, :TransposeFactorization) ?
                      LinearAlgebra.TransposeFactorization : Transpose

"""
    FactorizedDiffEqArrayOperator(F)

Like DiffEqArrayOperator, but stores a Factorization instead.

Supports left division and `ldiv!` when applied to an array.
"""
struct FactorizedDiffEqArrayOperator{T <: Number,
    FType <: Union{
        Factorization{T}, Diagonal{T}, Bidiagonal{T},
        AdjointFact{T, <:Factorization{T}},
        TransposeFact{T, <:Factorization{T}}
    }
} <: AbstractDiffEqLinearOperator{T}
    F::FType
end

function Base.convert(::Type{AbstractMatrix},
        L::FactorizedDiffEqArrayOperator{<:Any,
            <:Union{Factorization, AbstractMatrix
            }})
    convert(AbstractMatrix, L.F)
end
function Base.convert(::Type{AbstractMatrix},
        L::FactorizedDiffEqArrayOperator{<:Any, <:Union{Adjoint, AdjointFact}
        })
    adjoint(convert(AbstractMatrix, adjoint(L.F)))
end
function Base.convert(::Type{AbstractMatrix},
        L::FactorizedDiffEqArrayOperator{<:Any,
            <:Union{Transpose, TransposeFact}})
    transpose(convert(AbstractMatrix, transpose(L.F)))
end

function Base.Matrix(L::FactorizedDiffEqArrayOperator{<:Any,
        <:Union{Factorization, AbstractMatrix
        }})
    Matrix(L.F)
end
function Base.Matrix(L::FactorizedDiffEqArrayOperator{<:Any, <:Union{Adjoint, AdjointFact}})
    adjoint(Matrix(adjoint(L.F)))
end
function Base.Matrix(L::FactorizedDiffEqArrayOperator{<:Any,
        <:Union{Transpose, TransposeFact}})
    transpose(Matrix(transpose(L.F)))
end

Base.adjoint(L::FactorizedDiffEqArrayOperator) = FactorizedDiffEqArrayOperator(L.F')
Base.size(L::FactorizedDiffEqArrayOperator, args...) = size(L.F, args...)
function LinearAlgebra.ldiv!(Y::AbstractVecOrMat, L::FactorizedDiffEqArrayOperator,
        B::AbstractVecOrMat)
    ldiv!(Y, L.F, B)
end
LinearAlgebra.ldiv!(L::FactorizedDiffEqArrayOperator, B::AbstractVecOrMat) = ldiv!(L.F, B)
Base.:\(L::FactorizedDiffEqArrayOperator, x::AbstractVecOrMat) = L.F \ x
LinearAlgebra.issuccess(L::FactorizedDiffEqArrayOperator) = issuccess(L.F)

LinearAlgebra.ldiv!(y, L::FactorizedDiffEqArrayOperator, x) = ldiv!(y, L.F, x)
#isconstant(::FactorizedDiffEqArrayOperator) = true
has_ldiv(::FactorizedDiffEqArrayOperator) = true
has_ldiv!(::FactorizedDiffEqArrayOperator) = true

# The (u,p,t) and (du,u,p,t) interface
for T in [DiffEqScalar, DiffEqArrayOperator, FactorizedDiffEqArrayOperator, DiffEqIdentity]
    (L::T)(u, p, t) = (update_coefficients!(L, u, p, t); L * u)
    (L::T)(du, u, p, t) = (update_coefficients!(L, u, p, t); mul!(du, L, u))
end
