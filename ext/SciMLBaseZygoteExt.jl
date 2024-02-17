module SciMLBaseZygoteExt

using Zygote
using Zygote: @adjoint, pullback
import Zygote: literal_getproperty
using SciMLBase
using SciMLBase: ODESolution, remake, 
                 getobserved, build_solution, EnsembleSolution,
                 NonlinearSolution, AbstractTimeseriesSolution
using SymbolicIndexingInterface: symbolic_type, NotSymbolic, variable_index
using RecursiveArrayTools

# This method resolves the ambiguity with the pullback defined in
# RecursiveArrayToolsZygoteExt
# https://github.com/SciML/RecursiveArrayTools.jl/blob/d06ecb856f43bc5e37cbaf50e5f63c578bf3f1bd/ext/RecursiveArrayToolsZygoteExt.jl#L67
@adjoint function getindex(VA::ODESolution, i::Int, j::Int)
    function ODESolution_getindex_pullback(Δ)
        du = [m == j ? [i == k ? Δ : zero(VA.u[1][1]) for k in 1:length(VA.u[1])] :
              zero(VA.u[1]) for m in 1:length(VA.u)]
        dp = zero(VA.prob.p)
        dprob = remake(VA.prob, p = dp)
        du, dprob
        T = eltype(eltype(VA.u))
        N = length(VA.prob.p)
        Δ′ = ODESolution{T, N, typeof(du), Nothing, Nothing, typeof(VA.t),
            typeof(VA.k), typeof(dprob), typeof(VA.alg), typeof(VA.interp),
            typeof(VA.stats), typeof(VA.alg_choice)}(du, nothing, nothing,
            VA.t, VA.k, dprob, VA.alg, VA.interp, VA.dense, 0, VA.stats,
            VA.alg_choice, VA.retcode)
        (Δ′, nothing, nothing)
    end
    VA[i, j], ODESolution_getindex_pullback
end

@adjoint function getindex(VA::ODESolution, sym, j::Int)
    function ODESolution_getindex_pullback(Δ)
        i = symbolic_type(sym) != NotSymbolic() ? variable_index(VA, sym) : sym
        du, dprob = if i === nothing
            getter = getobserved(VA)
            grz = pullback(getter, sym, VA.u[j], VA.prob.p, VA.t[j])[2](Δ)
            du = [k == j ? grz[2] : zero(VA.u[1]) for k in 1:length(VA.u)]
            dp = grz[3] # pullback for p
            dprob = remake(VA.prob, p = dp)
            du, dprob
        else
            du = [m == j ? [i == k ? Δ : zero(VA.u[1][1]) for k in 1:length(VA.u[1])] :
                  zero(VA.u[1]) for m in 1:length(VA.u)]
            dp = zero(VA.prob.p)
            dprob = remake(VA.prob, p = dp)
            du, dprob
        end
        T = eltype(eltype(VA.u))
        N = length(VA.prob.p)
        Δ′ = ODESolution{T, N, typeof(du), Nothing, Nothing, typeof(VA.t),
            typeof(VA.k), typeof(dprob), typeof(VA.alg), typeof(VA.interp),
            typeof(VA.stats), typeof(VA.alg_choice)}(du, nothing, nothing,
            VA.t, VA.k, dprob, VA.alg, VA.interp, VA.dense, 0, VA.stats,
            VA.alg_choice, VA.retcode)
        (Δ′, nothing, nothing)
    end
    VA[sym, j], ODESolution_getindex_pullback
end

@adjoint function EnsembleSolution(sim, time, converged, stats)
    out = EnsembleSolution(sim, time, converged, stats)
    function EnsembleSolution_adjoint(p̄::AbstractArray{T, N}) where {T, N}
        arrarr = [[p̄[ntuple(x -> Colon(), Val(N - 2))..., j, i]
                   for j in 1:size(p̄)[end - 1]] for i in 1:size(p̄)[end]]
        (EnsembleSolution(arrarr, 0.0, true, stats), nothing, nothing, nothing)
    end
    function EnsembleSolution_adjoint(p̄::AbstractArray{<:AbstractArray, 1})
        (EnsembleSolution(p̄, 0.0, true, stats), nothing, nothing, nothing)
    end
    function EnsembleSolution_adjoint(p̄::RecursiveArrayTools.AbstractVectorOfArray)
        (EnsembleSolution(p̄, 0.0, true, stats), nothing, nothing, nothing)
    end
    function EnsembleSolution_adjoint(p̄::EnsembleSolution)
        (p̄, nothing, nothing, nothing)
    end
    out, EnsembleSolution_adjoint
end

@adjoint function getindex(VA::ODESolution, i::Int)
    function ODESolution_getindex_pullback(Δ)
        Δ′ = [(i == j ? Δ : Zygote.FillArrays.Fill(zero(eltype(x)), size(x)))
              for (x, j) in zip(VA.u, 1:length(VA))]
        (Δ′, nothing)
    end
    VA[:, i], ODESolution_getindex_pullback
end

@adjoint function Zygote.literal_getproperty(sim::EnsembleSolution,
    ::Val{:u})
    sim.u, p̄ -> (EnsembleSolution(p̄, 0.0, true, sim.stats),)
end

@adjoint function getindex(VA::ODESolution, sym)
    function ODESolution_getindex_pullback(Δ)
        i = symbolic_type(sym) != NotSymbolic() ? variable_index(VA, sym) : sym
        if i === nothing
            throw(error("Zygote AD of purely-symbolic slicing for observed quantities is not yet supported. Work around this by using `A[sym,i]` to access each element sequentially in the function being differentiated."))
        else
            Δ′ = [[i == k ? Δ[j] : zero(x[1]) for k in 1:length(x)]
                  for (x, j) in zip(VA.u, 1:length(VA))]
            (Δ′, nothing)
        end
    end
    VA[sym], ODESolution_getindex_pullback
end

@adjoint function ODESolution{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12
    }(u,
        args...) where {T1, T2, T3, T4, T5, T6, T7, T8,
        T9, T10, T11, T12}
        function ODESolutionAdjoint(ȳ)
            (ȳ, ntuple(_ -> nothing, length(args))...)
        end
    
        ODESolution{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12}(u, args...),
        ODESolutionAdjoint
end

@adjoint function SDEProblem{uType, tType, isinplace, P, NP, F, G, K, ND}(u,
    args...) where
    {uType, tType, isinplace, P, NP, F, G, K, ND}
    function SDESolutionAdjoint(ȳ)
        (ȳ, ntuple(_ -> nothing, length(args))...)
    end

    SDESolution{uType, tType, isinplace, P, NP, F, G, K, ND}(u, args...), SDESolutionAdjoint
end

@adjoint function NonlinearSolution{T, N, uType, R, P, A, O, uType2}(u,
    args...) where {
    T,
    N,
    uType,
    R,
    P,
    A,
    O,
    uType2,
}
    function NonlinearSolutionAdjoint(ȳ)
        (ȳ, ntuple(_ -> nothing, length(args))...)
    end
    NonlinearSolution{T, N, uType, R, P, A, O, uType2}(u, args...), NonlinearSolutionAdjoint
end

@adjoint function literal_getproperty(sol::AbstractTimeseriesSolution,
    ::Val{:u})
    function solu_adjoint(Δ)
        zerou = zero(sol.prob.u0)
        _Δ = @. ifelse(Δ === nothing, (zerou,), Δ)
        (build_solution(sol.prob, sol.alg, sol.t, _Δ),)
    end
    sol.u, solu_adjoint
end

@adjoint function literal_getproperty(sol::SciMLBase.AbstractNoTimeSolution,
    ::Val{:u})
    function solu_adjoint(Δ)
        zerou = zero(sol.prob.u0)
        _Δ = @. ifelse(Δ === nothing, zerou, Δ)
        (build_solution(sol.prob, sol.alg, _Δ, sol.resid),)
    end
    sol.u, solu_adjoint
end

@adjoint function literal_getproperty(sol::SciMLBase.OptimizationSolution,
    ::Val{:u})
    function solu_adjoint(Δ)
        zerou = zero(sol.u)
        _Δ = @. ifelse(Δ === nothing, zerou, Δ)
        (build_solution(sol.cache, sol.alg, _Δ, sol.objective),)
    end
    sol.u, solu_adjoint
end

function ∇tmap(cx, f, args...)
    ys_and_backs = SciMLBase.tmap((args...) -> Zygote._pullback(cx, f, args...), args...)
    if isempty(ys_and_backs)
        ys_and_backs, _ -> (NoTangent(), NoTangent())
    else
        ys, backs = Zygote.unzip(ys_and_backs)
        function ∇tmap_internal(Δ)
            Δf_and_args_zipped = SciMLBase.tmap((f, δ) -> f(δ), backs, Δ)
            Δf_and_args = Zygote.unzip(Δf_and_args_zipped)
            Δf = reduce(Zygote.accum, Δf_and_args[1])
            (Δf, Δf_and_args[2:end]...)
        end
        ys, ∇tmap_internal
    end
end

function ∇responsible_map(cx, f, args...)
    ys_and_backs = SciMLBase.responsible_map((args...) -> Zygote._pullback(cx, f, args...),
        args...)
    if isempty(ys_and_backs)
        ys_and_backs, _ -> (NoTangent(), NoTangent())
    else
        ys, backs = Zygote.unzip(ys_and_backs)
        ys,
        function ∇responsible_map_internal(Δ)
            # Apply pullbacks in reverse order. Needed for correctness if `f` is stateful.
            Δf_and_args_zipped = SciMLBase.responsible_map((f, δ) -> f(δ),
                Zygote._tryreverse(SciMLBase.responsible_map,
                    backs, Δ)...)
            Δf_and_args = Zygote.unzip(Zygote._tryreverse(SciMLBase.responsible_map,
                Δf_and_args_zipped))
            Δf = reduce(Zygote.accum, Δf_and_args[1])
            (Δf, Δf_and_args[2:end]...)
        end
    end
end

@adjoint function SciMLBase.tmap(f, args::Union{AbstractArray, Tuple}...)
    ∇tmap(__context__, f, args...)
end

@adjoint function SciMLBase.responsible_map(f,
    args::Union{AbstractArray, Tuple
    }...)
    ∇responsible_map(__context__, f, args...)
end

end
