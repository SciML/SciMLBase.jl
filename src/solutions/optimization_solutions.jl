abstract type AbstractOptimizationSolution{T, N} <: AbstractNoTimeSolution{T, N} end

"""
$(TYPEDEF)

Representation of the solution to an nonlinear optimization defined by an OptimizationProblem

## Fields

- `u`: the representation of the optimization's solution.
- `cache`: the `OptimizationCache` that was solved.
- `alg`: the algorithm type used by the solver.
- `original`: if the solver is wrapped from an alternative solver ecosystem, such as
  Optim.jl, then this is the original return from said solver library.
- `retcode`: the return code from the solver. Used to determine whether the solver solved
  successfully (`sol.retcode === ReturnCode.Success`), whether it terminated due to a user-defined
  callback (`sol.retcode === :Terminated`), or whether it exited due to an error. For more
  details, see the return code section of the DifferentialEquations.jl documentation.
"""
struct OptimizationSolution{T, N, uType, C, A, Tf, O} <: AbstractOptimizationSolution{T, N}
    u::uType # minimizer
    cache::C # optimization cache
    alg::A # algorithm
    minimum::Tf
    retcode::ReturnCode.T
    original::O # original output of the optimizer
end

function build_optimization_solution(cache,
                        alg, u, minimum;
                        retcode = ReturnCode.Default,
                        original = nothing,
                        kwargs...)
    T = eltype(eltype(u))
    N = ndims(u)

    OptimizationSolution{T, N, typeof(u), typeof(cache), typeof(alg),
                         typeof(minimum), typeof(original)}(u, cache, alg, minimum, retcode,
                                                            original)
end

function Base.show(io::IO, A::AbstractOptimizationSolution)
    println(io, string("retcode: ", A.retcode))
    print(io, "u: ")
    show(io, A.u)
    println(io)
    print(io, "Final objective value:     $(A.minimum)\n")
    return
end

Base.@propagate_inbounds function Base.getproperty(x::AbstractOptimizationSolution,
                                                   s::Symbol)
    if s === :minimizer
        return getfield(x, :u)
    end
    return getfield(x, s)
end

function Base.summary(io::IO, A::AbstractOptimizationSolution)
    type_color, no_color = get_colorizers(io)
    print(io,
          type_color, nameof(typeof(A)),
          no_color, " with uType ",
          type_color, eltype(A.u),
          no_color)
end

function observed(A::AbstractOptimizationSolution, sym)
    getobserved(A)(sym, A.u, A.cache.p)
end