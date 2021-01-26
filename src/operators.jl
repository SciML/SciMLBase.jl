### AbstractSciMLOperator Interface

#=
1. Function call and multiplication: L(du, u, p, t) for inplace and du = L(u, p, t) for
   out-of-place, meaning L*u and mul!(du, L, u).
2. If the operator is not a constant, update it with (u,p,t). A mutating form, i.e.
   update_coefficients!(A,u,p,t) that changes the internal coefficients, and a
   out-of-place form B = update_coefficients(A,u,p,t).
3. isconstant(A) trait for whether the operator is constant or not.
4. islinear(A) trait for whether the operator is linear or not.
=#

Base.eltype(L::AbstractSciMLOperator{T}) where T = T
update_coefficients!(L,u,p,t) = nothing
update_coefficients(L,u,p,t) = L

# Traits
isconstant(::AbstractSciMLOperator) = false
islinear(::AbstractSciMLOperator) = false
has_expmv!(L::AbstractSciMLOperator) = false # expmv!(v, L, t, u)
has_expmv(L::AbstractSciMLOperator) = false # v = exp(L, t, u)
has_exp(L::AbstractSciMLOperator) = false # v = exp(L, t)*u
has_mul(L::AbstractSciMLOperator) = true # du = L*u
has_mul!(L::AbstractSciMLOperator) = false # mul!(du, L, u)
has_ldiv(L::AbstractSciMLOperator) = false # du = L\u
has_ldiv!(L::AbstractSciMLOperator) = false # ldiv!(du, L, u)

### AbstractDiffEqLinearOperator Interface

#=
1. AbstractDiffEqLinearOperator <: AbstractSciMLOperator
2. Can absorb under multiplication by a scalar. In all algorithms things like
   dt*L show up all the time, so the linear operator must be able to absorb
   such constants.
4. isconstant(A) trait for whether the operator is constant or not.
5. Optional: diagonal, symmetric, etc traits from LinearMaps.jl.
6. Optional: exp(A). Required for simple exponential integration.
7. Optional: expmv(A,u,p,t) = exp(t*A)*u and expmv!(v,A::DiffEqOperator,u,p,t)
   Required for sparse-saving exponential integration.
8. Optional: factorizations. A_ldiv_B, factorize et. al. This is only required
   for algorithms which use the factorization of the operator (Crank-Nicholson),
   and only for when the default linear solve is used.
=#

# Extra standard assumptions
isconstant(::AbstractDiffEqLinearOperator) = true
islinear(o::AbstractDiffEqLinearOperator) = isconstant(o)

# Other ones from LinearMaps.jl
# Generic fallbacks
LinearAlgebra.exp(L::AbstractDiffEqLinearOperator,t) = exp(t*L)
has_exp(L::AbstractDiffEqLinearOperator) = true
expmv(L::AbstractDiffEqLinearOperator,u,p,t) = exp(L,t)*u
expmv!(v,L::AbstractDiffEqLinearOperator,u,p,t) = mul!(v,exp(L,t),u)
# Factorizations have no fallback and just error
