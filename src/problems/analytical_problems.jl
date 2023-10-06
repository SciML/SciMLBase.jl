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

TruncatedStacktraces.@truncate_stacktrace AnalyticalProblem 3 1 2

function AnalyticalProblem(f, u0, tspan, p = NullParameters(); kwargs...)
    iip = isinplace(f, 4)
    AnalyticalProblem{iip}(f, u0, tspan, p; kwargs...)
end

export AnalyticalProblem, AbstractAnalyticalProblem
