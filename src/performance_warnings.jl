
const PERFORMANCE_WARNINGS = Preferences.@load_preference("PerformanceWarnings", true)

should_warn_paramtype(p::AbstractArray) = !isconcretetype(eltype(p))
should_warn_paramtype(::Dict{K, V}) where {K, V} = !isconcretetype(V)
should_warn_paramtype(p::AbstractArray{<:AbstractArray}) = any(should_warn_paramtype, p)
should_warn_paramtype(p::Tuple) = any(should_warn_paramtype, p)
should_warn_paramtype(p::NamedTuple) = any(should_warn_paramtype, p)
should_warn_paramtype(p) = false

const WARN_PARAMTYPE_MESSAGE = """
Using arrays or dicts to store parameters of different types can hurt performance.
Consider using tuples instead.
"""

"""
    warn_paramtype(p, warn_performance=PERFORMANCE_WARNINGS)

Inspect the type of `p` and emit a warning if it could hurt
performance when used to hold problem parameters.

The warning can be turned off by setting `warn_performance` to `false`.
To turn it off globally within the active project you can execute the following code, or put it in your `startup.jl`.

```julia
using Preferences, UUIDs
set_preferences!(
    UUID("1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"), "PerformanceWarnings" => false)
```
"""
function warn_paramtype(p, warn_performance = PERFORMANCE_WARNINGS)
    if warn_performance && should_warn_paramtype(p)
        @warn WARN_PARAMTYPE_MESSAGE maxlog=1
    end
end
