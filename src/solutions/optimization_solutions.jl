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
    OptimizationStats(
        s1.iterations + s2.iterations, s1.time + s2.time, s1.fevals + s2.fevals,
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

    Base.depwarn(
        "`build_solution(prob::AbstractOptimizationProblem, args...; kwargs...)` is deprecated." *
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

function Base.getproperty(cache::SciMLBase.AbstractOptimizationCache, x::Symbol)
    if x in (:u0, :p) && has_reinit(cache)
        return getfield(cache.reinit_cache, x)
    end
    return getfield(cache, x)
end

function has_reinit(cache::SciMLBase.AbstractOptimizationCache)
    hasfield(typeof(cache), :reinit_cache)
end
function reinit!(cache::SciMLBase.AbstractOptimizationCache; p = missing,
        u0 = missing, interpret_symbolicmap = true)
    if p === missing && u0 === missing
        p, u0 = cache.p, cache.u0
    else # at least one of them has a value
        if p === missing
            p = cache.p
        end
        if u0 === missing
            u0 = cache.u0
        end
        isu0symbolic = eltype(u0) <: Pair && !isempty(u0)
        ispsymbolic = eltype(p) <: Pair && !isempty(p) && interpret_symbolicmap
        if isu0symbolic && !has_sys(cache.f)
            throw(ArgumentError("This cache does not support symbolic maps with" *
                                " remake, i.e. it does not have a symbolic origin. Please use `remke`" *
                                "with the `u0` keyword argument as a vector of values, paying attention to" *
                                "parameter order."))
        end
        if ispsymbolic && !has_sys(cache.f)
            throw(ArgumentError("This cache does not support symbolic maps with " *
                                "`remake`, i.e. it does not have a symbolic origin. Please use `remake`" *
                                "with the `p` keyword argument as a vector of values (paying attention to" *
                                "parameter order) or pass `interpret_symbolicmap = false` as a keyword argument"))
        end
        if isu0symbolic && ispsymbolic
            p, u0 = process_p_u0_symbolic(cache, p, u0)
        elseif isu0symbolic
            _, u0 = process_p_u0_symbolic(cache, cache.p, u0)
        elseif ispsymbolic
            p, _ = process_p_u0_symbolic(cache, p, cache.u0)
        end
    end

    cache.reinit_cache.p = p
    cache.reinit_cache.u0 = u0

    return cache
end

SymbolicIndexingInterface.parameter_values(x::AbstractOptimizationCache) = x.p
SymbolicIndexingInterface.symbolic_container(x::AbstractOptimizationCache) = x.f

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

function SymbolicIndexingInterface.parameter_values(x::AbstractOptimizationSolution)
    parameter_values(x.cache)
end
SymbolicIndexingInterface.symbolic_container(x::AbstractOptimizationSolution) = x.cache

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
        Base.depwarn(
            "`sol.prob` is deprecated. Use getters like `get_p` or `get_syms` on `sol` instead.",
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
