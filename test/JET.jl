using NonlinearSolve
using LinearSolve
using LinearAlgebra
using ADTypes
using JET
const LS = LinearSolve

function f(u, p)
    L, U = cholesky(p.Σ)
    rhs = (u .* u .- p.λ)
    linprob = LinearProblem(Matrix(L), rhs)
    alg = LS.GenericLUFactorization()
    sol = LinearSolve.solve(linprob, alg)
    return sol.u 
end

function minimize(λ=1.0)
    ps = (; λ, Σ=hermitianpart(rand(2,2) + 2*I))
    u₀ = rand(2)
    prob = NonlinearLeastSquaresProblem{false}(f, u₀, ps)
    autodiff = AutoForwardDiff(; chunksize=1)
    sol = solve(prob, SimpleTrustRegion(; autodiff))
    return sol.u
end

@test_opt minimize()
