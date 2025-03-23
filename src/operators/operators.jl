#
# Extra standard assumptions
isconstant(::AbstractDiffEqLinearOperator) = true
islinear(o::AbstractDiffEqLinearOperator) = isconstant(o)

# Other ones from LinearMaps.jl
# Generic fallbacks
LinearAlgebra.exp(L::AbstractDiffEqLinearOperator, t) = exp(t * L)
has_exp(L::AbstractDiffEqLinearOperator) = true
expmv(L::AbstractDiffEqLinearOperator, u, p, t) = exp(L, t) * u
expmv!(v, L::AbstractDiffEqLinearOperator, u, p, t) = mul!(v, exp(L, t), u)
