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

struct WeightedEnsembleProblem{T1 <: AbstractEnsembleProblem, T2 <: AbstractVector} <: AbstractEnsembleProblem
  ensembleprob::T1
  weights::T2
end
Base.propertynames(e::WeightedEnsembleProblem) = (Base.propertynames(getfield(e, :ensembleprob))..., :weights)
function Base.getproperty(e::WeightedEnsembleProblem, f::Symbol)
  f === :weights && return getfield(e, :weights)
  f === :ensembleprob && return getfield(e, :ensembleprob)
  return getproperty(getfield(e, :ensembleprob), f)
end
function WeightedEnsembleProblem(args...; weights, kwargs...)
  # TODO: allow skipping checks?
  @assert sum(weights) ≈ 1
  ep = EnsembleProblem(args...; kwargs...)
  @assert length(ep.prob) == length(weights)
  WeightedEnsembleProblem(ep, weights)
end
