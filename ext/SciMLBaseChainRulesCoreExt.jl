module SciMLBaseChainRulesCoreExt

using SciMLBase
import ChainRulesCore
import ChainRulesCore: NoTangent, @non_differentiable

function ChainRulesCore.rrule(config::ChainRulesCore.RuleConfig{
        >:ChainRulesCore.HasReverseMode,
    },
    ::typeof(getindex),
    VA::ODESolution,
    sym,
    j::Integer)
    function ODESolution_getindex_pullback(Δ)
        i = symbolic_type(sym) != NotSymbolic() ? sym_to_index(sym, VA) : sym
        if i === nothing
            getter = getobserved(VA)
            grz = rrule_via_ad(config, getter, sym, VA.u[j], VA.prob.p, VA.t[j])[2](Δ)
            du = [k == j ? grz[2] : zero(VA.u[1]) for k in 1:length(VA.u)]
            dp = grz[3] # pullback for p
            dprob = remake(VA.prob, p = dp)
            T = eltype(eltype(VA.u))
            N = length(VA.prob.p)
            Δ′ = ODESolution{T, N, typeof(du), Nothing, Nothing, Nothing, Nothing,
                typeof(dprob), Nothing, Nothing, Nothing, Nothing}(du, nothing,
                nothing, nothing, nothing, dprob, nothing, nothing,
                VA.dense, 0, nothing, nothing, VA.retcode)
            (NoTangent(), Δ′, NoTangent(), NoTangent())
        else
            du = [m == j ? [i == k ? Δ : zero(VA.u[1][1]) for k in 1:length(VA.u[1])] :
                  zero(VA.u[1]) for m in 1:length(VA.u)]
            dp = zero(VA.prob.p)
            dprob = remake(VA.prob, p = dp)
            Δ′ = ODESolution{
                T,
                N,
                typeof(du),
                Nothing,
                Nothing,
                typeof(VA.t),
                typeof(VA.k),
                typeof(dprob),
                typeof(VA.alg),
                typeof(VA.interp),
                typeof(VA.alg_choice),
                typeof(VA.stats),
            }(du,
                nothing,
                nothing,
                VA.t,
                VA.k,
                dprob,
                VA.alg,
                VA.interp,
                VA.dense,
                0,
                VA.stats,
                VA.alg_choice,
                VA.retcode)
            (NoTangent(), Δ′, NoTangent(), NoTangent())
        end
    end
    VA[sym, j], ODESolution_getindex_pullback
end

function ChainRulesCore.rrule(::typeof(getindex), VA::ODESolution, sym)
    function ODESolution_getindex_pullback(Δ)
        i = symbolic_type(sym) != NotSymbolic() ? sym_to_index(sym, VA) : sym
        if i === nothing
            throw(error("AD of purely-symbolic slicing for observed quantities is not yet supported. Work around this by using `A[sym,i]` to access each element sequentially in the function being differentiated."))
        else
            Δ′ = [[i == k ? Δ[j] : zero(x[1]) for k in 1:length(x)]
                  for (x, j) in zip(VA.u, 1:length(VA))]
            (NoTangent(), Δ′, NoTangent())
        end
    end
    VA[sym], ODESolution_getindex_pullback
end

function ChainRulesCore.rrule(::Type{ODEProblem}, args...; kwargs...)
    function ODEProblemAdjoint(ȳ)
        (NoTangent(), ȳ.f, ȳ.u0, ȳ.tspan, ȳ.p, ȳ.kwargs, ȳ.problem_type)
    end

    ODEProblem(args...; kwargs...), ODEProblemAdjoint
end

function ChainRulesCore.rrule(::Type{SDEProblem}, args...; kwargs...)
    function SDEProblemAdjoint(ȳ)
        (NoTangent(), ȳ.f, ȳ.g, ȳ.u0, ȳ.tspan, ȳ.p, ȳ.kwargs, ȳ.problem_type)
    end

    SDEProblem(args...; kwargs...), SDEProblemAdjoint
end

function ChainRulesCore.rrule(::Type{
        <:ODESolution{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10,
            T11, T12,
        }}, u,
    args...) where {T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
    T12}
    function ODESolutionAdjoint(ȳ)
        (NoTangent(), ȳ, ntuple(_ -> NoTangent(), length(args))...)
    end

    ODESolution{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12}(u, args...),
    ODESolutionAdjoint
end

function ChainRulesCore.rrule(::Type{
    <:ODESolution{uType, tType, isinplace, P, NP, F, G, K,
        ND,
    }}, u,
    args...) where {uType, tType, isinplace, P, NP, F, G, K, ND}
    function SDESolutionAdjoint(ȳ)
        (NoTangent(), ȳ, ntuple(_ -> NoTangent(), length(args))...)
    end

    SDESolution{uType, tType, isinplace, P, NP, F, G, K, ND}(u, args...), SDESolutionAdjoint
end

function ChainRulesCore.rrule(::SciMLBase.EnsembleSolution, sim, time, converged)
    out = EnsembleSolution(sim, time, converged)
    function EnsembleSolution_adjoint(p̄::AbstractArray{T, N}) where {T, N}
        arrarr = [[p̄[ntuple(x -> Colon(), Val(N - 2))..., j, i]
                   for j in 1:size(p̄)[end - 1]] for i in 1:size(p̄)[end]]
        (NoTangent(), EnsembleSolution(arrarr, 0.0, true), NoTangent(), NoTangent())
    end
    function EnsembleSolution_adjoint(p̄::EnsembleSolution)
        (NoTangent(), p̄, NoTangent(), NoTangent())
    end
    out, EnsembleSolution_adjoint
end

end