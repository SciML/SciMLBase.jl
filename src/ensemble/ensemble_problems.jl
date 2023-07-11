"""
$(TYPEDEF)
"""
struct EnsembleProblem{T, T2, T3, T4, T5} <: AbstractEnsembleProblem
    prob::T
    prob_func::T2
    output_func::T3
    reduction::T4
    u_init::T5
    safetycopy::Bool
end

DEFAULT_PROB_FUNC(prob, i, repeat) = prob
DEFAULT_OUTPUT_FUNC(sol, i) = (sol, false)
DEFAULT_REDUCTION(u, data, I) = append!(u, data), false
DEFAULT_VECTOR_PROB_FUNC(prob, i, repeat) = prob[i]
function EnsembleProblem(prob::AbstractVector{<:AbstractSciMLProblem}; kwargs...)
    # TODO: @invoke
    invoke(EnsembleProblem, Tuple{Any}, prob; prob_func=DEFAULT_VECTOR_PROB_FUNC, kwargs...)
end
function EnsembleProblem(prob;
    output_func = DEFAULT_OUTPUT_FUNC,
    prob_func = DEFAULT_PROB_FUNC,
    reduction = DEFAULT_REDUCTION,
    u_init = nothing,
    safetycopy = prob_func !== DEFAULT_PROB_FUNC)
    EnsembleProblem(prob, prob_func, output_func, reduction, u_init, safetycopy)
end

function EnsembleProblem(; prob,
    output_func = DEFAULT_OUTPUT_FUNC,
    prob_func = DEFAULT_PROB_FUNC,
    reduction = DEFAULT_REDUCTION,
    u_init = nothing, p = nothing,
    safetycopy = prob_func !== DEFAULT_PROB_FUNC)
    EnsembleProblem(prob, prob_func, output_func, reduction, u_init, safetycopy)
end
