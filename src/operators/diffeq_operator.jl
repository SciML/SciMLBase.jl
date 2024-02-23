"""
AffineDiffEqOperator{T} <: AbstractDiffEqOperator{T}

`Ex: (A₁(t) + ... + Aₙ(t))*u + B₁(t) + ... + Bₘ(t)`

AffineDiffEqOperator{T}(As,Bs,du_cache=nothing)

Takes in two tuples for split Affine DiffEqs

 1. update_coefficients! works by updating the coefficients of the component
    operators.
 2. Function calls L(u, p, t) and L(du, u, p, t) are fallbacks interpreted in this form.
    This will allow them to work directly in the nonlinear ODE solvers without
    modification.
 3. f(du, u, p, t) is only allowed if a du_cache is given
 4. B(t) can be Union{Number,AbstractArray}, in which case they are constants.
    Otherwise they are interpreted they are functions v=B(t) and B(v,t)

Solvers will see this operator from integrator.f and can interpret it by
checking the internals of As and Bs. For example, it can check isconstant(As[1])
etc.
"""
struct AffineDiffEqOperator{T, T1, T2, U} <: AbstractDiffEqOperator{T}
    As::T1
    Bs::T2
    du_cache::U
    function AffineDiffEqOperator{T}(As, Bs, du_cache = nothing) where {T}
        all([size(a) == size(As[1])
             for a in As]) || error("Operator sizes do not agree")
        new{T, typeof(As), typeof(Bs), typeof(du_cache)}(As, Bs, du_cache)
    end
end

Base.size(L::AffineDiffEqOperator) = size(L.As[1])

function (L::AffineDiffEqOperator)(u, p, t::Number)
    update_coefficients!(L, u, p, t)
    du = sum(A * u for A in L.As)
    for B in L.Bs
        if B isa Union{Number, AbstractArray}
            du .+= B
        else
            du .+= B(t)
        end
    end
    du
end

function (L::AffineDiffEqOperator)(du, u, p, t::Number)
    update_coefficients!(L, u, p, t)
    L.du_cache === nothing &&
        error("Can only use inplace AffineDiffEqOperator if du_cache is given.")
    du_cache = L.du_cache
    fill!(du, zero(first(du)))
    # TODO: Make type-stable via recursion
    for A in L.As
        mul!(du_cache, A, u)
        du .+= du_cache
    end
    for B in L.Bs
        if B isa Union{Number, AbstractArray}
            du .+= B
        else
            B(du_cache, t)
            du .+= du_cache
        end
    end
end

function update_coefficients!(L::AffineDiffEqOperator, u, p, t)
    # TODO: Make type-stable via recursion
    for A in L.As
        update_coefficients!(A, u, p, t)
    end
    for B in L.Bs
        update_coefficients!(B, u, p, t)
    end
end

@deprecate is_constant(L::AbstractDiffEqOperator) isconstant(L)

# Scaled operator (α * A)
struct DiffEqScaledOperator{T, F, OpType <: AbstractDiffEqLinearOperator{T}} <:
       AbstractDiffEqCompositeOperator{T}
    coeff::DiffEqScalar{T, F}
    op::OpType
end

# Recursive routines that use `getops`
function update_coefficients!(L::AbstractDiffEqCompositeOperator, u, p, t)
    for op in getops(L)
        update_coefficients!(op, u, p, t)
    end
    L
end

function Base.:*(α::DiffEqScalar{T, F}, L::AbstractDiffEqLinearOperator{T}) where {T, F}
    DiffEqScaledOperator(α, L)
end
function Base.:*(α::Number, L::AbstractDiffEqLinearOperator{T}) where {T}
    DiffEqScaledOperator(DiffEqScalar(convert(T, α)), L)
end
Base.:-(L::AbstractDiffEqLinearOperator{T}) where {T} = DiffEqScalar(-one(T)) * L
getops(L::DiffEqScaledOperator) = (L.coeff, L.op)
Base.Matrix(L::DiffEqScaledOperator) = L.coeff * Matrix(L.op)
function Base.convert(::Type{AbstractMatrix}, L::DiffEqScaledOperator)
    L.coeff * convert(AbstractMatrix, L.op)
end

Base.size(L::DiffEqScaledOperator, i::Integer) = size(L.op, i)
Base.size(L::DiffEqScaledOperator, args...) = size(L.op, args...)
LinearAlgebra.opnorm(L::DiffEqScaledOperator, p::Real = 2) = abs(L.coeff) * opnorm(L.op, p)
Base.getindex(L::DiffEqScaledOperator, i::Int) = L.coeff * L.op[i]
Base.getindex(L::DiffEqScaledOperator, I::Vararg{Int, N}) where {N} = L.coeff * L.op[I...]

Base.:*(L::DiffEqScaledOperator, x::AbstractVecOrMat) = L.coeff * (L.op * x)
Base.:*(L::DiffEqScaledOperator, x::AbstractArray) = L.coeff * (L.op * x)

Base.:*(x::AbstractVecOrMat, L::DiffEqScaledOperator) = (x * L.op) * L.coeff
Base.:*(x::AbstractArray, L::DiffEqScaledOperator) = (x * L.op) * L.coeff

function LinearAlgebra.mul!(r::AbstractVecOrMat, L::DiffEqScaledOperator,
        x::AbstractVecOrMat)
    mul!(r, L.op, x)
    r .= r * L.coeff
end
function LinearAlgebra.mul!(r::AbstractArray, L::DiffEqScaledOperator, x::AbstractArray)
    mul!(r, L.op, x)
    r .= r * L.coeff
end

function LinearAlgebra.mul!(r::AbstractVecOrMat, x::AbstractVecOrMat,
        L::DiffEqScaledOperator)
    mul!(r, x, L.op)
    r .= r * L.coeff
end
function LinearAlgebra.mul!(r::AbstractArray, x::AbstractArray, L::DiffEqScaledOperator)
    mul!(r, x, L.op)
    r .= r * L.coeff
end

Base.:/(L::DiffEqScaledOperator, x::AbstractVecOrMat) = L.coeff * (L.op / x)
Base.:/(L::DiffEqScaledOperator, x::AbstractArray) = L.coeff * (L.op / x)

Base.:/(x::AbstractVecOrMat, L::DiffEqScaledOperator) = 1 / L.coeff * (x / L.op)
Base.:/(x::AbstractArray, L::DiffEqScaledOperator) = 1 / L.coeff * (x / L.op)

Base.:\(L::DiffEqScaledOperator, x::AbstractVecOrMat) = 1 / L.coeff * (L.op \ x)
Base.:\(L::DiffEqScaledOperator, x::AbstractArray) = 1 / L.coeff * (L.op \ x)

Base.:\(x::AbstractVecOrMat, L::DiffEqScaledOperator) = L.coeff * (x \ L)
Base.:\(x::AbstractArray, L::DiffEqScaledOperator) = L.coeff * (x \ L)

for N in (2, 3)
    @eval begin
        function LinearAlgebra.mul!(Y::AbstractArray{T, $N},
                L::DiffEqScaledOperator{T},
                B::AbstractArray{T, $N}) where {T}
            LinearAlgebra.lmul!(Y, L.coeff, mul!(Y, L.op, B))
        end
    end
end

function LinearAlgebra.ldiv!(Y::AbstractVecOrMat, L::DiffEqScaledOperator,
        B::AbstractVecOrMat)
    lmul!(1 / L.coeff, ldiv!(Y, L.op, B))
end
function LinearAlgebra.ldiv!(Y::AbstractArray, L::DiffEqScaledOperator, B::AbstractArray)
    lmul!(1 / L.coeff, ldiv!(Y, L.op, B))
end

LinearAlgebra.factorize(L::DiffEqScaledOperator) = L.coeff * factorize(L.op)
for fact in (:lu, :lu!, :qr, :qr!, :cholesky, :cholesky!, :ldlt, :ldlt!,
    :bunchkaufman, :bunchkaufman!, :lq, :lq!, :svd, :svd!)
    @eval function LinearAlgebra.$fact(L::DiffEqScaledOperator, args...)
        L.coeff * fact(L.op, args...)
    end
end

isconstant(L::AbstractDiffEqCompositeOperator) = all(isconstant, getops(L))
