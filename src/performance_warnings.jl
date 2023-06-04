
const PERFORMANCE_WARNINGS = Preferences.@load_preference("PERFORMANCE_WARNINGS", true)

"""
    set_performance_warnings!(b::Bool)

Enable/disable performance warnings globally.
"""
function set_performance_warnings!(b::Bool)
    Preferences.@set_preferences!("PERFORMANCE_WARNINGS" => b)
    if b
        @info("Performance Warnings enabled. Restart Julia for this change to take effect.")
    else
        @info("Performance Warnings disabled. Restart Julia for this change to take effect.")
    end
end

should_warn_paramtype(p::AbstractArray) = !isconcretetype(eltype(p))
should_warn_paramtype(::Dict{K,V}) where {K,V} = !isconcretetype(V)
should_warn_paramtype(p::AbstractArray{<:AbstractArray}) = any(should_warn_paramtype, p)
should_warn_paramtype(p::Tuple) = any(should_warn_paramtype, p)
should_warn_paramtype(p::NamedTuple) = any(should_warn_paramtype, p)
should_warn_paramtype(p) = false

const WARN_PARAMTYPE_MESSAGE = """
Using arrays or dicts to store parameters of different types can hurt performance. \
Consider using tuples instead.
"""

"""
    warn_paramtype(p, warn_performance=PERFORMANCE_WARNINGS)

Inspect the type of `p` and emit a warning if it could hurt
performance when used to hold problem parameters.

The warning can be turned off by setting `warn_performance` to `false`.
To turn it off globally, call `set_performance_warnings!(false)`.
This will take effect after restarting Julia.
"""
function warn_paramtype(p, warn_performance = PERFORMANCE_WARNINGS)
    if warn_performance && should_warn_paramtype(p)
        @info(WARN_PARAMTYPE_MESSAGE)
    end
end
