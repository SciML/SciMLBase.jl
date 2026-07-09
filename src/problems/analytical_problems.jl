"""
$(TYPEDEF)

Problem wrapper for systems solved by an analytical function.

`AnalyticalProblem` stores the analytical function, initial state, time span,
parameters, and solver keyword arguments. The function follows the same
in-place or out-of-place convention as differential-equation functions, while
the concrete analytical solver decides how to evaluate it across `tspan`.

# Fields

$(TYPEDFIELDS)
"""
struct AnalyticalProblem{uType, tType, isinplace, P, F, K} <:
    AbstractAnalyticalProblem{uType, tType, isinplace}
    f::F
    u0::uType
    tspan::tType
    p::P
    kwargs::K
    @add_kwonly function AnalyticalProblem{iip}(
            f, u0, tspan, p = NullParameters();
            kwargs...
        ) where {iip}
        _u0 = prepare_initial_state(u0)
        _tspan = promote_tspan(tspan)
        warn_paramtype(p)
        new{
            typeof(_u0), typeof(_tspan), iip, typeof(p),
            typeof(f), typeof(kwargs),
        }(
            f,
            _u0,
            _tspan,
            p,
            kwargs
        )
    end
end

function AnalyticalProblem(f, u0, tspan, p = NullParameters(); kwargs...)
    iip = isinplace(f, 4)
    return AnalyticalProblem{iip}(f, u0, tspan, p; kwargs...)
end

export AnalyticalProblem, AbstractAnalyticalProblem

@doc doc"""
    AnalyticalAliasSpecifier(;alias_p = nothing, alias_f = nothing, alias_u0 = nothing, alias_du0 = nothing, alias_tstops = nothing, alias = nothing)

Control which `AnalyticalProblem` inputs and solver option arrays may be
aliased.

`alias_u0` controls the initial state, `alias_du0` controls an initial
derivative array when present, `alias_p` controls the parameter object,
`alias_f` controls the analytical function object, and `alias_tstops` controls
the `tstops` vector. A value of `nothing` delegates to the solver default. Set
`alias = true` or `alias = false` to apply the same policy to all fields.

### Keywords

* `alias_p::Union{Bool, Nothing}`: alias the parameter object.
* `alias_f::Union{Bool, Nothing}`: alias the analytical function object.
* `alias_u0::Union{Bool, Nothing}`: alias the `u0` array.
* `alias_du0::Union{Bool, Nothing}`: alias the `du0` array, when present.
* `alias_tstops::Union{Bool, Nothing}`: alias the `tstops` array.
* `alias::Union{Bool, Nothing}`: set every field of the `AnalyticalAliasSpecifier`.
"""
struct AnalyticalAliasSpecifier <: AbstractAliasSpecifier
    alias_p::Union{Bool, Nothing}
    alias_f::Union{Bool, Nothing}
    alias_u0::Union{Bool, Nothing}
    alias_du0::Union{Bool, Nothing}
    alias_tstops::Union{Bool, Nothing}

    function AnalyticalAliasSpecifier(;
            alias_p = nothing, alias_f = nothing, alias_u0 = nothing,
            alias_du0 = nothing, alias_tstops = nothing, alias = nothing
        )
        return if alias == true
            new(true, true, true, true, true)
        elseif alias == false
            new(false, false, false, false, false)
        elseif isnothing(alias)
            new(alias_p, alias_f, alias_u0, alias_du0, alias_tstops)
        end
    end
end
