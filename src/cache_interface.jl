get_p(cache::AbstractOptimizationCache) = cache.p
get_observed(cache::AbstractOptimizationCache) = cache.f.observed
get_syms(cache::AbstractOptimizationCache) = cache.f.syms
get_paramsyms(cache::AbstractOptimizationCache)= cache.f.paramsyms

has_observed(cache::AbstractOptimizationCache) = get_observed(cache) !== nothing
has_syms(cache::AbstractOptimizationCache) = get_syms(cache) !== nothing
has_paramsyms(cache::AbstractOptimizationCache) = get_paramsyms(cache) !== nothing