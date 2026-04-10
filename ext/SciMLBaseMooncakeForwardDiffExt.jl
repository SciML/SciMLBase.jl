module SciMLBaseMooncakeForwardDiffExt

using SciMLBase
using SciMLBase: ChainRulesOriginator, MooncakeOriginator, AbstractTimeseriesSolution,
    getobserved
using SymbolicIndexingInterface: symbolic_type, NotSymbolic, variable_index,
    parameter_values
import Mooncake
using Mooncake: CoDual, zero_fcodual, @is_primitive, MinimalCtx,
    NoPullback, NoFData, NoRData, fdata, zero_tangent
using ForwardDiff: ForwardDiff

# These rules override the state-only versions in SciMLBaseMooncakeExt with
# full observable support via ForwardDiff. They are dispatched via the same
# `@is_primitive` declaration so the more specific signature wins.

@is_primitive MinimalCtx Tuple{typeof(getindex), AbstractTimeseriesSolution, Any}
@is_primitive MinimalCtx Tuple{
    typeof(getindex), AbstractTimeseriesSolution, Any, Integer,
}

function Mooncake.rrule!!(
        ::CoDual{typeof(getindex)},
        sol::CoDual{<:AbstractTimeseriesSolution},
        sym::CoDual,
    )
    VA = sol.x
    s = sym.x
    y = VA[s]
    y_fdata = zero(y)
    sol_fdata = sol.dx

    function _scatter_pullback(::NoRData)
        i = symbolic_type(s) != NotSymbolic() ? variable_index(VA, s) : s
        u_fd = sol_fdata.data.u
        if i !== nothing
            for k in eachindex(VA.u)
                u_fd[k][i] += y_fdata[k]
            end
        else
            getter = getobserved(VA)
            p = parameter_values(VA.prob)
            for k in eachindex(VA.u)
                t_k = VA.t[k]
                grad_u = ForwardDiff.gradient(u -> getter(s, u, p, t_k), VA.u[k])
                @. u_fd[k] += y_fdata[k] * grad_u
            end
        end
        return (NoRData(), NoRData(), NoRData())
    end

    return CoDual(y, y_fdata), _scatter_pullback
end

function Mooncake.rrule!!(
        ::CoDual{typeof(getindex)},
        sol::CoDual{<:AbstractTimeseriesSolution},
        sym::CoDual,
        j::CoDual{<:Integer},
    )
    VA = sol.x
    s = sym.x
    jp = j.x
    y = VA[s, jp]
    sol_fdata = sol.dx

    function _scatter_pullback_indexed(dy)
        i = symbolic_type(s) != NotSymbolic() ? variable_index(VA, s) : s
        u_fd = sol_fdata.data.u
        if i !== nothing
            u_fd[jp][i] += dy
        else
            getter = getobserved(VA)
            p = parameter_values(VA.prob)
            t_jp = VA.t[jp]
            grad_u = ForwardDiff.gradient(u -> getter(s, u, p, t_jp), VA.u[jp])
            @. u_fd[jp] += dy * grad_u
        end
        return (NoRData(), NoRData(), NoRData(), NoRData())
    end

    return zero_fcodual(y), _scatter_pullback_indexed
end

end
