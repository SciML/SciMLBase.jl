abstract type AbstractOptimizationSolution{T, N} <: AbstractNoTimeSolution{T, N} end

"""
$(TYPEDEF)

Representation of the solution to an nonlinear optimization defined by an OptimizationProblem

## Fields

- `u`: the representation of the optimization's solution.
- `cache::AbstractOptimizationCache`: the optimization cache` that was solved.
- `alg`: the algorithm type used by the solver.
- `original`: if the solver is wrapped from an alternative solver ecosystem, such as
  Optim.jl, then this is the original return from said solver library.
- `retcode`: the return code from the solver. Used to determine whether the solver solved
  successfully (`sol.retcode === ReturnCode.Success`), whether it terminated due to a user-defined
  callback (`sol.retcode === :Terminated`), or whether it exited due to an error. For more
  details, see the return code section of the DifferentialEquations.jl documentation.
"""
struct OptimizationSolution{T, N, uType, C <: AbstractOptimizationCache, A, Tf, O} <:
       AbstractOptimizationSolution{T, N}
    u::uType # minimizer
    cache::C # optimization cache
    alg::A # algorithm
    minimum::Tf
    retcode::Symbol
    original::O # original output of the optimizer
end

function build_solution(cache::AbstractOptimizationCache,
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

"""
$(TYPEDEF)

Representation the default cache for an optimization problem defined by an `OptimizationProblem`.
"""
mutable struct DefaultOptimizationCache{F <: OptimizationFunction, P} <:
               AbstractOptimizationCache
    f::F
    p::P
end

# for compatibility
function build_solution(prob::AbstractOptimizationProblem,
                        alg, u, minimum;
                        retcode = ReturnCode.Default,
                        original = nothing,
                        kwargs...)
    T = eltype(eltype(u))
    N = ndims(u)

    Base.depwarn("`build_solution(prob::AbstractOptimizationProblem, args...; kwargs...)` is deprecated." *
                 " Consider implementing an `AbstractOptimizationCache` instead.",
                 "build_solution(prob::AbstractOptimizationProblem, args...; kwargs...)")

    cache = DefaultOptimizationCache(prob.f, prob.p)

    OptimizationSolution{T, N, typeof(u), typeof(cache), typeof(alg),
                         typeof(minimum), typeof(original)}(u, cache, alg, minimum, retcode,
                                                            original)
end

get_p(sol::OptimizationSolution) = sol.cache.p
get_observed(sol::OptimizationSolution) = sol.cache.f.observed
get_syms(sol::OptimizationSolution) = sol.cache.f.syms
get_paramsyms(sol::OptimizationSolution) = sol.cache.f.paramsyms

has_observed(sol::OptimizationSolution) = get_observed(sol) !== nothing
has_syms(sol::OptimizationSolution) = get_syms(sol) !== nothing
has_paramsyms(sol::OptimizationSolution) = get_paramsyms(sol) !== nothing

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
    elseif s == :prob
        Base.depwarn("`sol.prob` is deprecated. Use getters like `get_p` or `get_syms` on `sol` instead.",
                     "sol.prob")
        return getfield(x, :cache)
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
