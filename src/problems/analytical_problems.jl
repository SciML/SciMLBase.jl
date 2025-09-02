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
    @add_kwonly function AnalyticalProblem{iip}(f, u0, tspan, p = NullParameters();
            kwargs...) where {iip}
        _u0 = prepare_initial_state(u0)
        _tspan = promote_tspan(tspan)
        warn_paramtype(p)
        new{typeof(_u0), typeof(_tspan), iip, typeof(p),
            typeof(f), typeof(kwargs)}(f,
            _u0,
            _tspan,
            p,
            kwargs)
    end
end

function AnalyticalProblem(f, u0, tspan, p = NullParameters(); kwargs...)
    iip = isinplace(f, 4)
    AnalyticalProblem{iip}(f, u0, tspan, p; kwargs...)
end

export AnalyticalProblem, AbstractAnalyticalProblem

@doc doc"""
    AnalyticalAliasSpecifier(;alias_p = nothing, alias_f = nothing, alias_u0 = nothing, alias_du0 = nothing, alias_tstops = nothing, alias = nothing)

Holds information on what variables to alias
when solving an AnalyticalProblem. Conforms to the AbstractAliasSpecifier interface. 

When a keyword argument is `nothing`, the default behaviour of the solver is used.
    
### Keywords 
* `alias_p::Union{Bool, Nothing}`
* `alias_f::Union{Bool, Nothing}`
* `alias_u0::Union{Bool, Nothing}`: alias the u0 array.
* `alias_du0::Union{Bool, Nothing}`: alias the du0 array for DAEs. Defaults to false.
* `alias_tstops::Union{Bool, Nothing}`: alias the tstops array
* `alias::Union{Bool, Nothing}`: sets all fields of the `AnalyticalAliasSpecifier` to `alias`

"""
struct AnalyticalAliasSpecifier <: AbstractAliasSpecifier
    alias_p::Union{Bool, Nothing}
    alias_f::Union{Bool, Nothing}
    alias_u0::Union{Bool, Nothing}
    alias_du0::Union{Bool, Nothing}
    alias_tstops::Union{Bool, Nothing}

    function AnalyticalAliasSpecifier(;
            alias_p = nothing, alias_f = nothing, alias_u0 = nothing,
            alias_du0 = nothing, alias_tstops = nothing, alias = nothing)
        if alias == true
            new(true, true, true, true, true)
        elseif alias == false
            new(false, false, false, false, false)
        elseif isnothing(alias)
            new(alias_p, alias_f, alias_u0, alias_du0, alias_tstops)
        end
    end
end
