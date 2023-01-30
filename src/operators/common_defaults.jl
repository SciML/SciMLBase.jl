# The `update_coefficients!` interface

# Routines that use the AbstractMatrix representation
function Base.convert(::Type{AbstractArray}, L::AbstractDiffEqLinearOperator)
    convert(AbstractMatrix, L)
end
function Base.size(L::AbstractDiffEqLinearOperator, args...)
    size(convert(AbstractMatrix, L), args...)
end
function LinearAlgebra.opnorm(L::AbstractDiffEqLinearOperator, p::Real = 2)
    opnorm(convert(AbstractMatrix, L), p)
end
Base.@propagate_inbounds function Base.getindex(L::AbstractDiffEqLinearOperator,
                                                I::Vararg{Any, N}) where {N}
    convert(AbstractMatrix, L)[I...]
end
function Base.getindex(L::AbstractDiffEqLinearOperator, I::Vararg{Int, N}) where {N}
    convert(AbstractMatrix, L)[I...]
end
for op in (:*, :/, :\)
    @eval function Base.$op(L::AbstractDiffEqLinearOperator, x::AbstractArray)
        $op(convert(AbstractMatrix, L), x)
    end
    @eval function Base.$op(x::AbstractArray, L::AbstractDiffEqLinearOperator)
        $op(x, convert(AbstractMatrix, L))
    end
    @eval Base.$op(L::DiffEqArrayOperator, x::Number) = $op(convert(AbstractMatrix, L), x)
    @eval Base.$op(x::Number, L::DiffEqArrayOperator) = $op(x, convert(AbstractMatrix, L))
end
function LinearAlgebra.mul!(Y::AbstractArray, L::AbstractDiffEqLinearOperator,
                            B::AbstractArray)
    mul!(Y, convert(AbstractMatrix, L), B)
end
function LinearAlgebra.mul!(Y::AbstractArray, L::AbstractDiffEqLinearOperator,
                            B::AbstractArray, α::Number, β::Number)
    mul!(Y, convert(AbstractMatrix, L), B, α, β)
end
for pred in (:isreal, :issymmetric, :ishermitian, :isposdef)
    @eval function LinearAlgebra.$pred(L::AbstractDiffEqLinearOperator)
        $pred(convert(AbstractArray, L))
    end
end
for op in (:sum, :prod)
    @eval function LinearAlgebra.$op(L::AbstractDiffEqLinearOperator; kwargs...)
        $op(convert(AbstractArray, L); kwargs...)
    end
end
function LinearAlgebra.factorize(L::AbstractDiffEqLinearOperator)
    FactorizedDiffEqArrayOperator(factorize(convert(AbstractMatrix, L)))
end
for fact in (:lu, :lu!, :qr, :qr!, :cholesky, :cholesky!, :ldlt, :ldlt!,
             :bunchkaufman, :bunchkaufman!, :lq, :lq!, :svd, :svd!)
    @eval function LinearAlgebra.$fact(L::AbstractDiffEqLinearOperator, args...)
        FactorizedDiffEqArrayOperator($fact(convert(AbstractMatrix, L), args...))
    end
    @eval function LinearAlgebra.$fact(L::AbstractDiffEqLinearOperator; kwargs...)
        FactorizedDiffEqArrayOperator($fact(convert(AbstractMatrix, L); kwargs...))
    end
end

# Routines that use the full matrix representation
Base.Matrix(L::AbstractDiffEqLinearOperator) = Matrix(convert(AbstractMatrix, L))
LinearAlgebra.exp(L::AbstractDiffEqLinearOperator) = exp(Matrix(L))
