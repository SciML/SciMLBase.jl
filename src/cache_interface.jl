function get_p(::AbstractOptimizationCache)
    error("`get_p`: method has not been implemented for the cache")
end
function get_observed(::AbstractOptimizationCache)
    error("`get_observed`: method has not been implemented for the cache")
end
function get_syms(::AbstractOptimizationCache)
    error("`get_syms`: method has not been implemented for the cache")
end
function get_paramsyms(::AbstractOptimizationCache)
    error("`get_paramsyms`: method has not been implemented for the cache")
end

has_observed(cache::AbstractOptimizationCache) = get_observed(cache) !== nothing
has_syms(cache::AbstractOptimizationCache) = get_syms(cache) !== nothing
has_paramsyms(cache::AbstractOptimizationCache) = get_paramsyms(cache) !== nothing