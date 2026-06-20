module SciMLBaseZygoteExt

using Zygote
using Zygote: @adjoint, pullback
import Zygote: literal_getproperty
import ChainRulesCore
using SciMLBase
using SciMLBase: ODESolution, remake, ODEFunction,
    getobserved, build_solution, EnsembleSolution,
    NonlinearSolution, AbstractTimeseriesSolution
using SymbolicIndexingInterface: symbolic_type, NotSymbolic, variable_index, is_observed,
    observed, parameter_values, state_values, current_time
using RecursiveArrayTools
import SciMLStructures

# SciML problem types are mutable structs, so Zygote routes their field-access
# cotangents through the per-`Context` mutable-struct accumulator (`grad_mut`),
# keyed on the problem instance. That accumulator is shared across all pullback
# invocations of one context, so when several independent pullbacks read fields
# of the *same* problem (one per trajectory in the ensemble `tmap`/
# `responsible_map` adjoints below) and their structural tangents are then
# `accum`ed, each tangent carries a snapshot of the *running* accumulator and the
# earlier trajectories' contributions are double-counted. (Zygote's Ref-identity
# protocol that normally prevents this is broken as soon as the cotangent is
# materialized at a ChainRules boundary, which the per-trajectory solve rrules
# guarantee.) Treat problems as immutable for reverse-mode AD instead, mirroring
# the `getproperty(::NonlinearProblem)` rrule in SciMLBaseChainRulesCoreExt.
# Scoped to `AbstractDEProblem` so that `AbstractNonlinearProblem` keeps its
# existing behavior — its nested-AD path already produces partial-`NamedTuple`
# problem cotangents (the `getproperty` rrule in the ChainRulesCore ext), which
# this full-`NamedTuple` pullback would collide with in `Zygote.accum`.
@adjoint function Zygote.literal_getfield(
        prob::SciMLBase.AbstractDEProblem, ::Val{f}
    ) where {f}
    val = getfield(prob, f)
    function problem_literal_getfield_pullback(Δ)
        Zygote.accum_param(__context__, val, Δ) === nothing && return (nothing, nothing)
        ((; Zygote.nt_nothing(prob)..., Zygote.pair(Val(f), Δ, prob)...), nothing)
    end
    val, problem_literal_getfield_pullback
end

@adjoint function SciMLBase.remake(prob::ODEFunction; kw...)
    y = remake(prob; kw...)
    function odefunction_remake_back(Δ)
        (Δ,)
    end
    y, odefunction_remake_back
end

@adjoint function Base.getindex(VA::ODESolution, sym, j::Integer)
    res, pullback = ChainRulesCore.rrule(Zygote.ZygoteRuleConfig(), getindex, VA, sym, j)
    return res, Base.tail ∘ pullback
end

@adjoint function EnsembleSolution(sim, time, converged, stats)
    out = EnsembleSolution(sim, time, converged, stats)
    function EnsembleSolution_adjoint(p̄::AbstractArray{T, N}) where {T, N}
        arrarr = [
            [
                    p̄[ntuple(x -> Colon(), Val(N - 2))..., j, i]
                    for j in 1:size(p̄)[end - 1]
                ] for i in 1:size(p̄)[end]
        ]
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
    function EnsembleSolution_adjoint(p̄::NamedTuple)
        (p̄.u, nothing, nothing, nothing)
    end
    out, EnsembleSolution_adjoint
end

# `sol[i::Integer]`: under RecursiveArrayTools v4 `AbstractVectorOfArray`
# subtypes `AbstractArray`, so linear integer indexing returns the i-th
# scalar element in column-major order over the underlying state-by-time
# layout (i.e. `VA[CartesianIndices(size(VA))[i]]`), NOT the i-th timestep
# vector. A dedicated adjoint is still needed here to keep dispatch from
# falling through to the broader `Base.getindex(VA::ODESolution, sym)`
# rule below (which would misinterpret `i` as a state-variable index;
# #1325). The pullback scatters the scalar cotangent into the matching
# slot of `VA.u`.
@adjoint function Base.getindex(VA::ODESolution, i::Integer)
    inds = Tuple(CartesianIndices(size(VA))[i])
    front_inds = Base.front(inds)
    step_idx = last(inds)
    y = VA.u[step_idx][front_inds...]
    function ODESolution_scalar_pullback(Δ)
        Δ′ = map(enumerate(VA.u)) do (k, x)
            if k == step_idx
                δu = zero(x)
                δu[front_inds...] = Δ
                δu
            else
                Zygote.FillArrays.Fill(zero(eltype(x)), size(x))
            end
        end
        return (Δ′, nothing)
    end
    return y, ODESolution_scalar_pullback
end

# `sol[I::CartesianIndex]`: under RecursiveArrayTools v4 an `ODESolution` is a
# 2D `AbstractArray`, so `eachindex(sol)` yields `CartesianIndex`es and
# `sol[CartesianIndex(state, step)]` returns the corresponding scalar. Without a
# dedicated adjoint this falls through to `getindex(VA::ODESolution, sym)`,
# which mis-treats the `CartesianIndex` as a state index and calls
# `view(VA, ::CartesianIndex, :)` — three indices into a 2D parent — failing
# with "number of indices (3) must match the parent dimensionality (2)". The
# cotangent is returned as a zeroed `ODESolution` (matching the input, as the
# `sym` adjoint below does) with the scalar `Δ` scattered into the indexed slot,
# so it has the shape `DifferentiationInterface`/`Zygote` expect for `sol`.
@adjoint function Base.getindex(VA::ODESolution, I::CartesianIndex)
    inds = Tuple(I)
    front_inds = Base.front(inds)
    step_idx = last(inds)
    y = VA.u[step_idx][front_inds...]
    function ODESolution_cartesian_pullback(Δ)
        dVA = recursivecopy(VA)
        recursivefill!(dVA, zero(eltype(dVA)))
        dVA.u[step_idx][front_inds...] = Δ
        return (dVA, nothing)
    end
    return y, ODESolution_cartesian_pullback
end

@adjoint function Base.getindex(VA::ODESolution, sym)
    function ODESolution_getindex_pullback(Δ)
        i = symbolic_type(sym) != NotSymbolic() ? variable_index(VA, sym) : sym
        if is_observed(VA, sym)
            f = observed(VA, sym)
            p = parameter_values(VA)
            u = state_values(VA)
            t = current_time(VA)
            y, back = Zygote.pullback(u, p) do u, p
                f.(u, Ref(p), t)
            end
            gs = back(Δ)
            (u = gs[1], prob = (p = gs[2],)), nothing
        elseif i === nothing
            throw(error("Zygote AD of purely-symbolic slicing for observed quantities is not yet supported. Work around this by using `A[sym,i]` to access each element sequentially in the function being differentiated."))
        else
            VA = recursivecopy(VA)
            recursivefill!(VA, zero(eltype(VA)))
            v = view(VA, i, ntuple(_ -> :, ndims(VA) - 1)...)
            copyto!(v, Δ)
            (VA, nothing)
        end
    end
    VA[sym], ODESolution_getindex_pullback
end

function obs_grads(VA, sym, obs_idx, Δ)
    y, back = Zygote.pullback(VA) do sol
        getindex.(Ref(sol), sym[obs_idx])
    end
    Δreduced = reduce(hcat, Δ)
    Δobs = eachrow(Δreduced[obs_idx, :])
    return back(Δobs)
end

function obs_grads2(VA::SciMLBase.NonlinearSolution, sym, obs_idx, Δ)
    y, back = Zygote.pullback(VA) do sol
        getindex.(Ref(sol), sym[obs_idx])
    end
    Δobs = Δ[obs_idx, :]
    return back(Δobs)
end

function obs_grads(VA, sym, ::Nothing, Δ)
    return Zygote.nt_nothing(VA)
end

function not_obs_grads(VA::ODESolution{T}, sym, not_obss_idx, i, Δ) where {T}
    Δ′ = map(enumerate(VA.u)) do (t_idx, us)
        map(enumerate(us)) do (u_idx, u)
            if u_idx in i
                idx = findfirst(isequal(u_idx), i)
                Δ[t_idx][idx]
            else
                zero(T)
            end
        end
    end

    return Δ′
end

@adjoint function Base.getindex(
        VA::ODESolution{T}, sym::Union{Tuple, AbstractVector}
    ) where {T}
    function ODESolution_getindex_pullback(Δ)
        sym = sym isa Tuple ? collect(sym) : sym
        i = map(x -> symbolic_type(x) != NotSymbolic() ? variable_index(VA, x) : x, sym)

        obs_idx = findall(s -> is_observed(VA, s), sym)
        not_obs_idx = setdiff(1:length(sym), obs_idx)

        gs_obs = obs_grads(VA, sym, isempty(obs_idx) ? nothing : obs_idx, Δ)
        gs_not_obs = not_obs_grads(VA, sym, not_obs_idx, i, Δ)

        a = Zygote.accum(gs_obs[1], (u = gs_not_obs,))

        (a, nothing)
    end
    VA[sym], ODESolution_getindex_pullback
end

# `sol[syms, :]` returns the same per-timestep values of the selected states as
# `sol[syms]` (the trailing `:` selects all timesteps), so it reuses the
# vector-symbol adjoint above. Without this method the call has no `@adjoint`
# and falls through to ChainRules' generic array `getindex` rule, which tries to
# `fill!` a `Matrix{Vector{Float64}}` zero array with the scalar `false` and
# errors with "Cannot `convert` an object of type Bool to ...".
@adjoint function Base.getindex(
        VA::ODESolution{T}, sym::Union{Tuple, AbstractVector}, ::Colon
    ) where {T}
    y, pullback = Zygote.pullback(getindex, VA, sym)
    function ODESolution_getindex_colon_pullback(Δ)
        dVA, dsym = pullback(Δ)
        return (dVA, dsym, nothing)
    end
    return y, ODESolution_getindex_colon_pullback
end

@adjoint function Base.getindex(VA::SciMLBase.AbstractNonlinearSolution, sym)
    function NonlinearSolution_getindex_pullback(Δ)
        i = symbolic_type(sym) != NotSymbolic() ? variable_index(VA, sym) : sym
        if is_observed(VA, sym)
            f = observed(VA, sym)
            p = parameter_values(VA)
            u = state_values(VA)
            _, back = Zygote.pullback(u, p) do u, p
                f.f_oop(u, p)
            end
            gs = back(Δ)
            ((u = gs[1], prob = (p = gs[2],)), nothing)
        elseif i === nothing
            throw(error("Zygote AD of purely-symbolic slicing for observed quantities is not yet supported. Work around this by using `A[sym,i]` to access each element sequentially in the function being differentiated."))
        elseif i isa Int && VA.u isa Number
            (Δ, nothing)
        else
            VA = recursivecopy(VA)
            recursivefill!(VA, zero(eltype(VA)))
            v = view(VA, i, ntuple(_ -> :, ndims(VA) - 1)...)
            copyto!(v, Δ)
            (VA, nothing)
        end
    end
    VA[sym], NonlinearSolution_getindex_pullback
end

@adjoint function ODESolution{
        T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    }(
        u,
        args...
    ) where {
        T1, T2, T3, T4, T5, T6, T7, T8,
        T9, T10, T11, T12, T13, T14, T15,
    }
    function ODESolutionAdjoint(ȳ)
        (ȳ, ntuple(_ -> nothing, length(args))...)
    end

    ODESolution{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15}(
            u, args...
        ),
        ODESolutionAdjoint
end

@adjoint function SDEProblem{uType, tType, isinplace, P, NP, F, G, K, ND}(
        u,
        args...
    ) where
    {uType, tType, isinplace, P, NP, F, G, K, ND}
    function SDEProblemAdjoint(ȳ)
        (ȳ, ntuple(_ -> nothing, length(args))...)
    end

    SDEProblem{uType, tType, isinplace, P, NP, F, G, K, ND}(u, args...), SDEProblemAdjoint
end

@adjoint function NonlinearSolution{T, N, uType, R, P, A, O, uType2}(
        u,
        args...
    ) where {
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

@adjoint function literal_getproperty(sol::SciMLBase.LinearSolution, ::Val{:u})
    function solu_adjoint(Δ)
        zerou = zero(sol.u)
        _Δ = @. ifelse(Δ === nothing, zerou, Δ)
        (SciMLBase.build_linear_solution(sol.alg, _Δ, sol.resid, sol.cache),)
    end
    sol.u, solu_adjoint
end

@adjoint function Base.getindex(sol::SciMLBase.LinearSolution, i::Integer)
    function LinearSolution_getindex_pullback(Δ)
        du = zero(sol.u)
        du[i] = Δ
        (SciMLBase.build_linear_solution(sol.alg, du, sol.resid, sol.cache), nothing)
    end
    sol[i], LinearSolution_getindex_pullback
end

@adjoint function literal_getproperty(
        sol::SciMLBase.OptimizationSolution,
        ::Val{:u}
    )
    function solu_adjoint(Δ)
        zerou = zero(sol.u)
        _Δ = @. ifelse(Δ === nothing, zerou, Δ)
        (build_solution(sol.cache, sol.alg, _Δ, sol.objective),)
    end
    sol.u, solu_adjoint
end

# Under RecursiveArrayTools v4, `AbstractVectorOfArray` (including
# `EnsembleSolution` and `VectorOfArray`) is an `AbstractArray` whose iteration
# yields scalars in column-major order. Cotangents for `tmap`/`responsible_map`
# outputs can arrive wrapped in those containers (e.g. from the
# `EnsembleSolution` constructor adjoint above), so they must be unwrapped to
# the plain vector of per-element tangents before zipping them with the
# pullbacks.
function unwrap_map_cotangent(Δ)
    while Δ isa RecursiveArrayTools.AbstractVectorOfArray
        Δ = Δ.u
    end
    return Δ
end

function ∇tmap(cx, f, args...)
    ys_and_backs = SciMLBase.tmap((args...) -> Zygote._pullback(cx, f, args...), args...)
    return if isempty(ys_and_backs)
        ys_and_backs, _ -> (NoTangent(), NoTangent())
    else
        ys, backs = Zygote.unzip(ys_and_backs)
        function ∇tmap_internal(Δ)
            Δ = unwrap_map_cotangent(Δ)
            Δf_and_args_zipped = SciMLBase.tmap((f, δ) -> f(δ), backs, Δ)
            Δf_and_args = Zygote.unzip(Δf_and_args_zipped)
            Δf = reduce(Zygote.accum, Δf_and_args[1])
            return (Δf, Δf_and_args[2:end]...)
        end
        ys, ∇tmap_internal
    end
end

function ∇responsible_map(cx, f, args...)
    ys_and_backs = SciMLBase.responsible_map(
        (args...) -> Zygote._pullback(cx, f, args...),
        args...
    )
    return if isempty(ys_and_backs)
        ys_and_backs, _ -> (NoTangent(), NoTangent())
    else
        ys, backs = Zygote.unzip(ys_and_backs)
        ys,
            function ∇responsible_map_internal(Δ)
                Δ = unwrap_map_cotangent(Δ)
                # Apply pullbacks in reverse order. Needed for correctness if `f` is stateful.
                Δf_and_args_zipped = SciMLBase.responsible_map(
                    (f, δ) -> f(δ),
                    Zygote._tryreverse(
                        SciMLBase.responsible_map,
                        backs, Δ
                    )...
                )
                Δf_and_args = Zygote.unzip(
                    Zygote._tryreverse(
                        SciMLBase.responsible_map,
                        Δf_and_args_zipped
                    )
                )
                Δf = reduce(Zygote.accum, Δf_and_args[1])
                return (Δf, Δf_and_args[2:end]...)
        end
    end
end

@adjoint function SciMLBase.tmap(f, args::Union{AbstractArray, Tuple}...)
    ∇tmap(__context__, f, args...)
end

@adjoint function SciMLBase.responsible_map(
        f,
        args::Union{
            AbstractArray, Tuple,
        }...
    )
    ∇responsible_map(__context__, f, args...)
end

end
