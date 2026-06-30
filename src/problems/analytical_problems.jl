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
    function AnalyticalAliasSpecifier(;
            alias_p = nothing, alias_f = nothing, alias_u0 = nothing,
            alias_du0 = nothing, alias_tstops = nothing, alias = nothing
        )
        alias === true && return new{true, true, true, true, true}()
        alias === false && return new{false, false, false, false, false}()
        return new{alias_p, alias_f, alias_u0, alias_du0, alias_tstops}()
    end
end

function Base.getproperty(
        ::AnalyticalAliasSpecifier{P, F, U0, DU0, TS}, s::Symbol
    ) where {P, F, U0, DU0, TS}
    s === :alias_p && return P
    s === :alias_f && return F
    s === :alias_u0 && return U0
    s === :alias_du0 && return DU0
    s === :alias_tstops && return TS
    throw(ArgumentError("AnalyticalAliasSpecifier has no field $s"))
end
Base.propertynames(::AnalyticalAliasSpecifier) =
    (:alias_p, :alias_f, :alias_u0, :alias_du0, :alias_tstops)
