"""
$(TYPEDEF)

Stores the optimization run's statistics that is returned 
in the `stats` field of the `OptimizationResult`. 

## Fields
- `iterations`: number of iterations
- `time`: time taken to run the solver
- `fevals`: number of function evaluations
- `gevals`: number of gradient evaluations
- `hevals`: number of hessian evaluations

Default values for all the field are set to 0 and hence even when 
you might expect non-zero values due to unavilability of the information 
from the solver it would be 0.
"""
struct OptimizationStats
    iterations::Int
    time::Float64
    fevals::Int
    gevals::Int
    hevals::Int
end

function OptimizationStats(; iterations = 0, time = 0.0, fevals = 0, gevals = 0, hevals = 0)
    OptimizationStats(iterations, time, fevals, gevals, hevals)
end

function Base.merge(s1::OptimizationStats, s2::OptimizationStats)
    OptimizationStats(s1.iterations + s2.iterations, s1.time + s2.time, s1.fevals + s2.fevals,
        s1.gevals + s2.gevals, s1.hevals + s2.hevals)
end

"""
$(TYPEDEF)

Representation of the solution to a non-linear optimization defined by an OptimizationProblem

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
- `stats`: statistics of the solver, such as the number of function evaluations required.
"""
struct OptimizationSolution{T, N, uType, C <: AbstractOptimizationCache, A, OV, O, ST} <:
       AbstractOptimizationSolution{T, N}
    u::uType # minimizer
    cache::C # optimization cache
    alg::A # algorithm
    objective::OV
    retcode::ReturnCode.T
    original::O # original output of the optimizer
    stats::ST
end

function build_solution(cache::AbstractOptimizationCache,
    alg, u, objective;
    retcode = ReturnCode.Default,
    original = nothing,
    stats = nothing,
    kwargs...)
    T = eltype(eltype(u))
    N = ndims(u)

    #Backwords compatibility, remove ASAP
    retcode = symbol_to_ReturnCode(retcode)

    OptimizationSolution{T, N, typeof(u), typeof(cache), typeof(alg),
        typeof(objective), typeof(original), typeof(stats)}(u, cache, 
        alg, objective, retcode, original, stats)
end

TruncatedStacktraces.@truncate_stacktrace OptimizationSolution 1 2

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
        typeof(objective), typeof(original)}(u, cache, alg, objective,
        retcode,
        original)
end

get_p(sol::OptimizationSolution) = sol.cache.p
get_observed(sol::OptimizationSolution) = sol.cache.f.observed
get_syms(sol::OptimizationSolution) = variable_symbols(sol.cache.f)
get_paramsyms(sol::OptimizationSolution) = parameter_symbols(sol.cache.f)
has_observed(sol::OptimizationSolution) = get_observed(sol) !== nothing
has_syms(sol::OptimizationSolution) = !isempty(variable_symbols(sol.cache.f))
has_paramsyms(sol::OptimizationSolution) = !isempty(parameter_symbols(sol.cache.f))

function Base.show(io::IO, A::AbstractOptimizationSolution)
    println(io, string("retcode: ", A.retcode))
    print(io, "u: ")
    show(io, A.u)
    println(io)
    print(io, "Final objective value:     $(A.objective)\n")
    return
end

SymbolicIndexingInterface.parameter_values(x::AbstractOptimizationSolution) = x.cache.p
SymbolicIndexingInterface.symbolic_container(x::AbstractOptimizationSolution) = x.cache.f

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
    elseif s === :ps
        return ParameterIndexingProxy(x)
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
