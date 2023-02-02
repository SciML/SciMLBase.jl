#
#Base.eltype(::Type{AbstractSciMLOperator{T}}) where {T} = T
#Base.eltype(::AbstractSciMLOperator{T}) where {T} = T
#update_coefficients!(L, u, p, t) = nothing
#update_coefficients(L, u, p, t) = L

# Traits
#isconstant(::AbstractSciMLOperator) = false # SciMLOperators/interface.jl:121
#islinear(::AbstractSciMLOperator) = false
#has_adjoint(L::AbstractSciMLOperator) = false
#has_expmv!(L::AbstractSciMLOperator) = false
#has_expmv(L::AbstractSciMLOperator) = false
#has_exp(L::AbstractSciMLOperator) = false
#has_mul(L::AbstractSciMLOperator) = true
#has_mul!(L::AbstractSciMLOperator) = false
#has_ldiv(L::AbstractSciMLOperator) = false
#has_ldiv!(L::AbstractSciMLOperator) = false

### AbstractDiffEqLinearOperator Interface

# Extra standard assumptions
isconstant(::AbstractDiffEqLinearOperator) = true
islinear(o::AbstractDiffEqLinearOperator) = isconstant(o)
### VP
islinear(L) = false
isconstant(L) = false
###

#isconstant(::AbstractMatrix) = true
#islinear(::AbstractMatrix) = true
#has_adjoint(::AbstractMatrix) = true
#has_mul(::AbstractMatrix) = true
#has_mul!(::AbstractMatrix) = true
#has_ldiv(::AbstractMatrix) = true
#has_ldiv!(::AbstractMatrix) = false
#has_ldiv!(::Union{Diagonal, Factorization}) = true

# Other ones from LinearMaps.jl
# Generic fallbacks
LinearAlgebra.exp(L::AbstractDiffEqLinearOperator, t) = exp(t * L)
has_exp(L::AbstractDiffEqLinearOperator) = true
expmv(L::AbstractDiffEqLinearOperator, u, p, t) = exp(L, t) * u
expmv!(v, L::AbstractDiffEqLinearOperator, u, p, t) = mul!(v, exp(L, t), u)
# Factorizations have no fallback and just error
