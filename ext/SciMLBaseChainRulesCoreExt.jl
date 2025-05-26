module SciMLBaseChainRulesCoreExt

using SciMLBase
using SciMLBase: getobserved
import ChainRulesCore
import ChainRulesCore: NoTangent, @non_differentiable, zero_tangent, rrule_via_ad
using SymbolicIndexingInterface

function ChainRulesCore.rrule(
        config::ChainRulesCore.RuleConfig{
            >:ChainRulesCore.HasReverseMode,
        },
        ::typeof(getindex),
        VA::ODESolution,
        sym,
        j::Integer)
    function ODESolution_getindex_pullback(Δ)
        i = symbolic_type(sym) != NotSymbolic() ? variable_index(VA, sym) : sym
        du, dprob = if i === nothing
            getter = getobserved(VA)
            grz = rrule_via_ad(config, getter, sym, VA.u[j], VA.prob.p, VA.t[j])[2](Δ)
            du = [k == j ? grz[3] : zero(VA.u[1]) for k in 1:length(VA.u)]
            dp = grz[4] # pullback for p
            if dp == NoTangent()
                dp = zero_tangent(parameter_values(VA.prob))
            end
            dprob = remake(VA.prob, p = dp)
            du, dprob
        else
            du = [m == j ? [i == k ? Δ : zero(VA.u[1][1]) for k in 1:length(VA.u[1])] :
                  zero(VA.u[1]) for m in 1:length(VA.u)]
            dp = zero_tangent(VA.prob.p)
            dprob = remake(VA.prob, p = dp)
            du, dprob
        end
        T = eltype(eltype(du))
        N = ndims(eltype(du)) + 1
        Δ′ = ODESolution{T, N}(du, nothing, nothing, VA.t, VA.k, nothing, dprob,
            VA.alg, VA.interp, VA.dense, 0, VA.stats, VA.alg_choice, VA.retcode)
        (NoTangent(), Δ′, NoTangent(), NoTangent())
    end
    VA[sym, j], ODESolution_getindex_pullback
end

function ChainRulesCore.rrule(::typeof(getindex), VA::ODESolution, sym)
    function ODESolution_getindex_pullback(Δ)
        i = symbolic_type(sym) != NotSymbolic() ? variable_index(VA, sym) : sym
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
        @show "some con"
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

function ChainRulesCore.rrule(
        ::Type{
            <:ODESolution{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10,
            T11, T12, T13, T14, T15, T16
        }}, u,
        args...) where {T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
        T12, T13, T14, T15, T16}
    function ODESolutionAdjoint(ȳ)
        (NoTangent(), ȳ, ntuple(_ -> NoTangent(), length(args))...)
    end

    ODESolution{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16}(u, args...),
    ODESolutionAdjoint
end

function ChainRulesCore.rrule(
        ::Type{
            <:RODESolution{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10,
            T11, T12, T13, T14
        }}, u,
        args...) where {T1, T2, T3, T4, T5, T6, T7, T8, T9, T10,
        T11, T12, T13, T14}
    function RODESolutionAdjoint(ȳ)
        (NoTangent(), ȳ, ntuple(_ -> NoTangent(), length(args))...)
    end

    RODESolution{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10,
    T11, T12, T13, T14}(u, args...),
    RODESolutionAdjoint
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

function ChainRulesCore.rrule(::Type{SciMLBase.IntervalNonlinearProblem}, args...; kwargs...)
    function IntervalNonlinearProblemAdjoint(ȳ)
        (NoTangent(), ȳ.f, ȳ.tspan, ȳ.p, ȳ.kwargs, ȳ.problem_type)
    end

    SciMLBase.IntervalNonlinearProblem(args...; kwargs...), IntervalNonlinearProblemAdjoint
end

end