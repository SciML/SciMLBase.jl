### Concrete Types

"""
$(TYPEDEF)

Representation of the solution to an stochastic differential equation defined by an SDEProblem,
or of a random ordinary differential equation defined by an RODEProblem.

## DESolution Interface

For more information on interacting with `DESolution` types, check out the Solution Handling
page of the DifferentialEquations.jl documentation.

https://docs.sciml.ai/DiffEqDocs/stable/basics/solution/

## Fields

- `u`: the representation of the SDE or RODE solution. Given as an array of solutions, where `u[i]`
  corresponds to the solution at time `t[i]`. It is recommended in most cases one does not
  access `sol.u` directly and instead use the array interface described in the Solution
  Handling page of the DifferentialEquations.jl documentation.
- `t`: the time points corresponding to the saved values of the ODE solution.
- `W`: the representation of the saved noise process from the solution. See [the Noise Processes
  page of the DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/features/noise_process/).
  Note that this noise is only saved in full if `save_noise=true` in the solver.
- `prob`: the original SDEProblem/RODEProblem that was solved.
- `alg`: the algorithm type used by the solver.
- `stats`: statistics of the solver, such as the number of function evaluations required,
  number of Jacobians computed, and more.
- `retcode`: the return code from the solver. Used to determine whether the solver solved
  successfully, whether it terminated early due to a user-defined callback, or whether it
  exited due to an error. For more details, see
  [the return code documentation](https://docs.sciml.ai/SciMLBase/stable/interfaces/Solutions/#retcodes).
"""
struct RODESolution{T, N, uType, uType2, DType, tType, randType, P, A, IType, S,
    AC <: Union{Nothing, Vector{Int}}, V} <:
       AbstractRODESolution{T, N, uType}
    u::uType
    u_analytic::uType2
    errors::DType
    t::tType
    W::randType
    prob::P
    alg::A
    interp::IType
    dense::Bool
    tslocation::Int
    stats::S
    alg_choice::AC
    retcode::ReturnCode.T
    seed::UInt64
    saved_subsystem::V
end

function ConstructionBase.constructorof(::Type{O}) where {T, N, O <: RODESolution{T, N}}
    RODESolution{T, N}
end

function ConstructionBase.setproperties(sol::RODESolution, patch::NamedTuple)
    u = get(patch, :u, sol.u)
    N = u === nothing ? 2 : ndims(eltype(u)) + 1
    T = eltype(eltype(u))
    patch = merge(getproperties(sol), patch)
    return RODESolution{
        T, N, typeof(patch.u), typeof(patch.u_analytic), typeof(patch.errors),
        typeof(patch.t), typeof(patch.W), typeof(patch.prob), typeof(patch.alg), typeof(patch.interp),
        typeof(patch.stats), typeof(patch.alg_choice), typeof(patch.saved_subsystem)}(
        patch.u, patch.u_analytic, patch.errors, patch.t, patch.W,
        patch.prob, patch.alg, patch.interp, patch.dense, patch.tslocation, patch.stats,
        patch.alg_choice, patch.retcode, patch.seed, patch.saved_subsystem)
end

Base.@propagate_inbounds function Base.getproperty(x::AbstractRODESolution, s::Symbol)
    if s === :destats
        Base.depwarn("`sol.destats` is deprecated. Use `sol.stats` instead.", "sol.destats")
        return getfield(x, :stats)
    elseif s === :ps
        return ParameterIndexingProxy(x)
    end
    return getfield(x, s)
end

function (sol::RODESolution)(t, ::Type{deriv} = Val{0}; idxs = nothing,
        continuity = :left) where {deriv}
    sol(t, deriv, idxs, continuity)
end
function (sol::RODESolution)(v, t, ::Type{deriv} = Val{0}; idxs = nothing,
        continuity = :left) where {deriv}
    sol.interp(v, t, idxs, deriv, sol.prob.p, continuity)
end

function build_solution(prob::Union{AbstractRODEProblem, AbstractSDDEProblem},
        alg, t, u; W = nothing, timeseries_errors = length(u) > 2,
        dense = false, dense_errors = dense, calculate_error = true,
        interp = LinearInterpolation(t, u),
        retcode = ReturnCode.Default,
        alg_choice = nothing,
        seed = UInt64(0), destats = missing, stats = nothing,
        saved_subsystem = nothing, kwargs...)
    T = eltype(eltype(u))
    if prob.u0 === nothing
        N = 2
    else
        N = ndims(eltype(u)) + 1
    end

    if prob.f isa Tuple
        f = prob.f[1]
    else
        f = prob.f
    end

    if !ismissing(destats)
        msg = "`destats` kwarg has been deprecated in favor of `stats`"
        if stats !== nothing
            msg *= " `stats` kwarg is also provided, ignoring `destats` kwarg."
        else
            stats = destats
        end
        Base.depwarn(msg, :build_solution)
    end

    if has_analytic(f)
        u_analytic = Vector{typeof(prob.u0)}()
        errors = Dict{Symbol, real(eltype(prob.u0))}()
        sol = RODESolution{T, N, typeof(u), typeof(u_analytic), typeof(errors), typeof(t),
            typeof(W),
            typeof(prob), typeof(alg), typeof(interp), typeof(stats),
            typeof(alg_choice), typeof(saved_subsystem)}(u,
            u_analytic,
            errors,
            t, W,
            prob,
            alg,
            interp,
            dense,
            0,
            stats,
            alg_choice,
            retcode,
            seed,
            saved_subsystem)

        if calculate_error
            calculate_solution_errors!(sol; timeseries_errors = timeseries_errors,
                dense_errors = dense_errors)
        end

        return sol
    else
        return RODESolution{T, N, typeof(u), Nothing, Nothing, typeof(t),
            typeof(W), typeof(prob), typeof(alg), typeof(interp),
            typeof(stats), typeof(alg_choice), typeof(saved_subsystem)}(
            u, nothing, nothing, t, W,
            prob, alg, interp,
            dense, 0, stats,
            alg_choice, retcode, seed, saved_subsystem)
    end
end

function calculate_solution_errors!(sol::AbstractRODESolution; fill_uanalytic = true,
        timeseries_errors = true, dense_errors = true)
    if sol.prob.f isa Tuple
        f = sol.prob.f[1]
    else
        f = sol.prob.f
    end

    if fill_uanalytic
        empty!(sol.u_analytic)
        if f isa RODEFunction && f.analytic_full == true
            f.analytic(sol)
        elseif sol.W isa AbstractDiffEqArray{T, N, nothing} where {T, N}
            for i in 1:length(sol)
                push!(sol.u_analytic,
                    f.analytic(sol.prob.u0, sol.prob.p, sol.t[i], first(sol.W(sol.t[i]))))
            end
        else
            for i in 1:length(sol)
                push!(sol.u_analytic,
                    f.analytic(sol.prob.u0, sol.prob.p, sol.t[i], sol.W[:, i]))
            end
        end
    end

    if !isempty(sol.u_analytic)
        sol.errors[:final] = norm(recursive_mean(abs.(sol.u[end] - sol.u_analytic[end])))
        if timeseries_errors
            sol.errors[:l∞] = norm(maximum(vecvecapply((x) -> abs.(x),
                sol.u - sol.u_analytic)))
            sol.errors[:l2] = norm(sqrt(recursive_mean(vecvecapply((x) -> float.(x) .^ 2,
                sol.u - sol.u_analytic))))
        end
        if dense_errors
            densetimes = collect(range(sol.t[1], stop = sol.t[end], length = 100))
            interp_u = sol(densetimes)
            interp_analytic = [f.analytic(sol.u[1], sol.prob.p, t, sol.W(t)[1])
                               for t in densetimes]
            sol.errors[:L∞] = norm(maximum(vecvecapply((x) -> abs.(x),
                interp_u - interp_analytic)))
            sol.errors[:L2] = norm(sqrt(recursive_mean(vecvecapply((x) -> float.(x) .^ 2,
                interp_u -
                interp_analytic))))
        end
    end
end

function build_solution(sol::AbstractRODESolution, u_analytic, errors)
    @reset sol.u_analytic = u_analytic
    return @set sol.errors = errors
end

function solution_new_retcode(sol::AbstractRODESolution, retcode)
    return @set sol.retcode = retcode
end

function solution_new_tslocation(sol::AbstractRODESolution, tslocation)
    return @set sol.tslocation = tslocation
end

function solution_slice(sol::AbstractRODESolution{T, N}, I) where {T, N}
    @reset sol.u = sol.u[I]
    @reset sol.u_analytic = sol.u_analytic === nothing ? nothing : sol.u_analytic[I]
    @reset sol.t = sol.t[I]
    return @set sol.dense = false
end

function sensitivity_solution(sol::AbstractRODESolution, u, t)
    T = eltype(eltype(u))

    # handle save_idxs
    u0 = first(u)
    if u0 isa Number
        N = 1
    else
        N = length((size(u0)..., length(u)))
    end

    interp = enable_interpolation_sensitivitymode(sol.interp)
    @reset sol.u = u
    @reset sol.t = t isa Vector ? t : collect(t)
    return @set sol.interp = interp
end
