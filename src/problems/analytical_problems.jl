"""
$(TYPEDEF)
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
    AnalyticalAliasSpecifier{P, F, U0, DU0, TS}(;alias_p = nothing, alias_f = nothing, alias_u0 = nothing, alias_du0 = nothing, alias_tstops = nothing, alias = nothing)

Holds information on what variables to alias
when solving an AnalyticalProblem. Conforms to the AbstractAliasSpecifier interface. 

When a keyword argument is `nothing`, the default behaviour of the solver is used.
    
### Keywords 
* `alias_p`
* `alias_f`
* `alias_u0`: alias the `u0` array.
* `alias_du0`: alias the `du0` array for DAEs. Defaults to `false`.
* `alias_tstops`: alias the `tstops` array
* `alias`: sets all fields of the `AnalyticalAliasSpecifier` to `alias`

"""
struct AnalyticalAliasSpecifier{P, F, U0, DU0, TS} <: AbstractAliasSpecifier
    alias_p::P
    alias_f::F
    alias_u0::U0
    alias_du0::DU0
    alias_tstops::TS

    function AnalyticalAliasSpecifier(;
            alias_p = nothing, alias_f = nothing, alias_u0 = nothing,
            alias_du0 = nothing, alias_tstops = nothing, alias = nothing
        )
        return if alias == true
            new{Bool, Bool, Bool, Bool, Bool}(true, true, true, true, true)
        elseif alias == false
            new{Bool, Bool, Bool, Bool, Bool}(false, false, false, false, false)
        elseif isnothing(alias)
            new{
                typeof(alias_p), typeof(alias_f), typeof(alias_u0),
                typeof(alias_du0), typeof(alias_tstops)
            }(alias_p, alias_f, alias_u0, alias_du0, alias_tstops)
        end
    end
end
