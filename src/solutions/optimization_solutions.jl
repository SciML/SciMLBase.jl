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
                         typeof(minimum), typeof(original)}(
                         u, prob, alg, minimum, retcode, original)
end

function Base.show(io::IO, A::AbstractOptimizationSolution)
    println(io,string("retcode: ",A.retcode))
    print(io,"u: ")
    show(io, A.u)
    println(io)
    print(io,"Final objective value:     $(A.minimum)\n")
    return
end

Base.@propagate_inbounds function Base.getproperty(x::AbstractOptimizationSolution,s::Symbol)
    if s === :minimizer
        return getfield(x,:u)
    end
    return getfield(x,s)
end

Base.summary(A::AbstractOptimizationSolution) = string(
                      TYPE_COLOR, nameof(typeof(A)),
                      NO_COLOR, " with uType ",
                      TYPE_COLOR, eltype(A.u),
                      NO_COLOR)
