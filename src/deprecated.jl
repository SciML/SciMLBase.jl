#
# Deprecated Quadrature things
const AbstractQuadratureProblem = AbstractIntegralProblem
const AbstractQuadratureAlgorithm = AbstractIntegralAlgorithm
const AbstractQuadratureSolution = AbstractIntegralSolution

# Deprecated High Level things
# All downstream uses need to be removed before removing

const DEAlgorithm = AbstractDEAlgorithm
const SciMLAlgorithm = AbstractSciMLAlgorithm
const DEProblem = AbstractDEProblem
const DEAlgorithm = AbstractDEAlgorithm
const DESolution = AbstractSciMLSolution
const SciMLSolution = AbstractSciMLSolution

# Deprecated operator interface
#=
const AbstractDiffEqOperator = AbstractSciMLOperator
const AbstractDiffEqLinearOperator = AbstractSciMLOperator
const AbstractDiffEqCompositeOperator = AbstractSciMLOperator

const DiffEqScaledOperator = ScaledOperator
function DiffEqScaledOperator(args...; kwargs...)
    @warn "SciMLBase.DiffEqScaledOperator is deprecated.
    Use SciMLOperators.ScaledOperator instead"

    ScaledOperator(args...; kwargs...)
end

const FactorizedDiffEqArrayOperator = InvertedOperator
function FactorizedDiffEqArrayOperator(args...; kwargs...)
    @warn "SciMLBase.FactorizedDiffEqArrayOperator is deprecated.
    Use SciMLOperators.InvertedOperator instead"

    InvertedOperator(args...; kwargs...)
end

const DiffEqIdentity = IdentityOperator
function DiffEqIdentity(u)
    @warn "SciMLBase.DiffEqIdentity is deprecated.
    Use SciMLOperators.IdentityOperator instead"

    IdentityOperator{size(u, 1)}()
end

const DiffEqScalar = SciMLOperators.ScalarOperator
function DiffEqScalar(args...; kwargs...)
    @warn "SciMLBase.DiffEqScalar is deprecated.
    Use SciMLOperators.ScalarOperator instead"

    ScalarOperator(args...; kwargs...)
end

const AffineDiffEqOperator = SciMLOperators.AffineOperator
function AffineDiffEqOperator{T}(As, bs, cache = nothing) where {T}
    @warn "SciMLBase.AffineDiffEqOperator is deprecated.
    Use SciMLOperators.AffineOperator instead"

    bs = isempty(bs) ? (zeros(Bool, size(As[1], 1)),) : bs

    all([size(a) == size(As[1]) for a in As]) || error("Operator sizes do not agree")
    all([size(b) == size(bs[1]) for b in bs]) || error("Vector sizes do not agree")

    A = AddedOperator(As)
    b = sum(bs)
	B = IdentityOperator{size(b, 1)}()

    AffineOperator(A, B, b)
end

const DiffEqArrayOperator = SciMLOperators.MatrixOperator
function DiffEqArrayOperator(args...; kwargs...)
    @warn "SciMLBase.DiffEqArrayOperator is deprecated.
        Use SciMLOperators.MatrixOperator instead"

    MatrixOperator(args...; kwargs...)
end
=#
#
