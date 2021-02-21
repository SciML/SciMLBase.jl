abstract type AbstractOptimizationSolution{T, N} <: AbstractNoTimeSolution{T, N} end

struct OptimizationSolution{T, N, uType, P, A, Tf, O} <: AbstractOptimizationSolution{T, N}
    u::uType # minimizer
    prob::P # optimization problem
    alg::A # algorithm
    minimum::Tf
    retcode::Symbol
    original::O # original output of the optimizer
end

function build_solution(prob::AbstractOptimizationProblem,
                        alg, u, minimum;
                        retcode = :Default,
                        original = nothing,
                        kwargs...)

    T = eltype(eltype(u))
    N = ndims(u)

    OptimizationSolution{T, N, typeof(u), typeof(prob), typeof(alg),
                         typeof(minimum), typeof(original)}
                         (u, prob, alg, minimum, retcode, original)
end

function Base.show(io::IO, A::AbstractOptimizationSolution)

    @printf io "\n * Status: %s\n\n" A.retcode === :Success ? "success" : "failure"
    @printf io " * Candidate solution\n"
    @printf io "    Final objective value:     %e\n" A.minimum
    @printf io "\n"
    @printf io " * Found with\n"
    @printf io "    Algorithm:     %s\n" A.alg
    return
end
