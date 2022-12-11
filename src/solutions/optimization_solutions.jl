abstract type AbstractOptimizationSolution{T, N} <: AbstractNoTimeSolution{T, N} end

"""
$(TYPEDEF)

Representation of the solution to an non-linear optimization defined by an OptimizationProblem

## Fields

- `u`: the representation of the optimization's solution.
- `cache::AbstractOptimizationCache`: the optimization cache` that was solved.
- `alg`: the algorithm type used by the solver.
- `objective`: Objective value of the solution
- `retcode`: the return code from the solver. Used to determine whether the solver solved
  successfully or whether it exited due to an error. For more details, see 
  [the return code documentation](https://docs.sciml.ai/SciMLBase/stable/interfaces/Solutions/#retcodes).
- `original`: if the solver is wrapped from an alternative solver ecosystem, such as
  Optim.jl, then this is the original return from said solver library.
- `solve_time`: Solve time from the solver in Seconds
"""
struct OptimizationSolution{T, N, uType, C <: AbstractOptimizationCache, A, OV, O, S} <:
       AbstractOptimizationSolution{T, N}
    u::uType # minimizer
    cache::C # optimization cache
    alg::A # algorithm
    objective::OV
    retcode::ReturnCode.T
    original::O # original output of the optimizer
    solve_time::S # [s] solve time from the solver
end

function build_solution(cache::AbstractOptimizationCache,
                        alg, u, objective;
                        retcode = ReturnCode.Default,
                        original = nothing,
                        solve_time = nothing,
                        kwargs...)
    T = eltype(eltype(u))
    N = ndims(u)

    #Backwords compatibility, remove ASAP
    retcode = symbol_to_ReturnCode(retcode)

    OptimizationSolution{T, N, typeof(u), typeof(cache), typeof(alg),
                         typeof(objective), typeof(original), typeof(solve_time)}(u, cache,
                                                                                alg,
                                                                                objective,
                                                                                retcode,
                                                                                original,
                                                                                solve_time)
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
                        alg, u, objective;
                        retcode = ReturnCode.Default,
                        original = nothing,
                        kwargs...)
    T = eltype(eltype(u))
    N = ndims(u)

    Base.depwarn("`build_solution(prob::AbstractOptimizationProblem, args...; kwargs...)` is deprecated." *
                 " Consider implementing an `AbstractOptimizationCache` instead.",
                 "build_solution(prob::AbstractOptimizationProblem, args...; kwargs...)")

    cache = DefaultOptimizationCache(prob.f, prob.p)

    #Backwords compatibility, remove ASAP
    retcode = symbol_to_ReturnCode(retcode)

    OptimizationSolution{T, N, typeof(u), typeof(cache), typeof(alg),
                         typeof(objective), typeof(original)}(u, cache, alg, objective, retcode,
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
    print(io, "Final objective value:     $(A.objective)\n")
    return
end

Base.@propagate_inbounds function Base.getproperty(x::AbstractOptimizationSolution,
                                                   s::Symbol)
    if s === :minimizer
        Base.depwarn("`sol.minimizer` is deprecated. Use `sol.u` instead.",
                     "sol.minimizer")
        return getfield(x, :u)
    elseif s === :minimum
        Base.depwarn("`sol.minimum` is deprecated. Use `sol.objective` instead.",
                     "sol.minimum")
        return getfield(x, :objective)
    elseif s === :prob
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
