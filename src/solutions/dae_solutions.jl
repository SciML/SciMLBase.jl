"""
$(TYPEDEF)

Representation of the solution to an differential-algebraic equation defined by an DAEProblem.

## DESolution Interface

For more information on interacting with `DESolution` types, check out the Solution Handling
page of the DifferentialEquations.jl documentation.

https://docs.sciml.ai/DiffEqDocs/stable/basics/solution/

## Fields

- `u`: the representation of the DAE solution. Given as an array of solutions, where `u[i]`
  corresponds to the solution at time `t[i]`. It is recommended in most cases one does not
  access `sol.u` directly and instead use the array interface described in the Solution
  Handling page of the DifferentialEquations.jl documentation.
- `du`: the representation of the derivatives of the DAE solution.
- `t`: the time points corresponding to the saved values of the DAE solution.
- `prob`: the original DAEProblem that was solved.
- `alg`: the algorithm type used by the solver.
- `stats`: statistics of the solver, such as the number of function evaluations required,
  number of Jacobians computed, and more.
- `retcode`: the return code from the solver. Used to determine whether the solver solved
  successfully, whether it terminated early due to a user-defined callback, or whether it
  exited due to an error. For more details, see
  [the return code documentation](https://docs.sciml.ai/SciMLBase/stable/interfaces/Solutions/#retcodes).
"""
struct DAESolution{T, N, uType, duType, uType2, DType, tType, P, A, ID, S, rateType, V} <:
       AbstractDAESolution{T, N, uType}
    u::uType
    du::duType
    u_analytic::uType2
    errors::DType
    t::tType
    k::rateType
    prob::P
    alg::A
    interp::ID
    dense::Bool
    tslocation::Int
    stats::S
    retcode::ReturnCode.T
    saved_subsystem::V
end

function DAESolution{T, N}(u, du, u_analytic, errors, t, k, prob, alg, interp, dense,
        tslocation, stats, retcode, saved_subsystem) where {T, N}
    return DAESolution{T, N, typeof(u), typeof(du), typeof(u_analytic), typeof(errors),
        typeof(t), typeof(prob), typeof(alg), typeof(interp), typeof(stats), typeof(k),
        typeof(saved_subsystem)}(
        u, du, u_analytic, errors, t, k, prob, alg, interp, dense, tslocation, stats,
        retcode, saved_subsystem
    )
end

function ConstructionBase.constructorof(::Type{O}) where {T, N, O <: DAESolution{T, N}}
    DAESolution{T, N}
end

function ConstructionBase.setproperties(sol::DAESolution, patch::NamedTuple)
    u = get(patch, :u, sol.u)
    N = u === nothing ? 2 : ndims(eltype(u)) + 1
    T = eltype(eltype(u))
    patch = merge(getproperties(sol), patch)
    return DAESolution{T, N}(patch.u, patch.du, patch.u_analytic, patch.errors, patch.t,
        patch.k, patch.prob, patch.alg, patch.interp, patch.dense, patch.tslocation,
        patch.stats, patch.retcode, patch.saved_subsystem)
end

Base.@propagate_inbounds function Base.getproperty(x::AbstractDAESolution, s::Symbol)
    if s === :destats
        Base.depwarn("`sol.destats` is deprecated. Use `sol.stats` instead.", "sol.destats")
        return getfield(x, :stats)
    elseif s === :ps
        return ParameterIndexingProxy(x)
    end
    return getfield(x, s)
end

function build_solution(prob::AbstractDAEProblem, alg, t, u, du = nothing;
        timeseries_errors = length(u) > 2,
        dense = false,
        dense_errors = dense,
        calculate_error = true,
        k = nothing,
        interp = du === nothing ? LinearInterpolation(t, u) :
                 HermiteInterpolation(t, u, du),
        retcode = ReturnCode.Default,
        destats = missing,
        stats = nothing,
        saved_subsystem = nothing,
        kwargs...)
    T = eltype(eltype(u))

    if prob.u0 === nothing
        N = 2
    else
        N = ndims(eltype(u)) + 1
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
    if has_analytic(prob.f)
        u_analytic = Vector{typeof(prob.u0)}()
        errors = Dict{Symbol, real(eltype(prob.u0))}()

        sol = DAESolution{T, N, typeof(u), typeof(du), typeof(u_analytic), typeof(errors),
            typeof(t), typeof(prob), typeof(alg), typeof(interp), typeof(stats), typeof(k),
            typeof(saved_subsystem)}(
            u,
            du,
            u_analytic,
            errors,
            t,
            k,
            prob,
            alg,
            interp,
            dense,
            0,
            stats,
            retcode,
            saved_subsystem)

        if calculate_error
            calculate_solution_errors!(sol; timeseries_errors = timeseries_errors,
                dense_errors = dense_errors)
        end
        sol
    else
        DAESolution{T, N, typeof(u), typeof(du), Nothing, Nothing, typeof(t),
            typeof(prob), typeof(alg), typeof(interp), typeof(stats), typeof(k),
            typeof(saved_subsystem)}(
            u, du,
            nothing,
            nothing, t, k,
            prob, alg,
            interp,
            dense, 0,
            stats,
            retcode,
            saved_subsystem)
    end
end

function calculate_solution_errors!(sol::AbstractDAESolution;
        fill_uanalytic = true, timeseries_errors = true,
        dense_errors = true)
    prob = sol.prob
    f = prob.f

    if fill_uanalytic
        for i in 1:size(sol.u, 1)
            push!(sol.u_analytic, f.analytic(prob.du0, prob.u0, prob.p, sol.t[i]))
        end
    end

    save_everystep = length(sol.u) > 2
    if !isempty(sol.u_analytic)
        sol.errors[:final] = norm(recursive_mean(abs.(sol.u[end] - sol.u_analytic[end])))

        if save_everystep && timeseries_errors
            sol.errors[:l∞] = norm(maximum(vecvecapply(x -> abs.(x),
                sol.u - sol.u_analytic)))
            sol.errors[:l2] = norm(sqrt(recursive_mean(vecvecapply(x -> float.(x) .^ 2,
                sol.u - sol.u_analytic))))
            if sol.dense && dense_errors
                densetimes = collect(range(sol.t[1]; stop = sol.t[end], length = 100))
                interp_u = sol(densetimes)
                interp_analytic = VectorOfArray([f.analytic(prob.du0, prob.u0, prob.p, t)
                                                 for t in densetimes])
                sol.errors[:L∞] = norm(maximum(vecvecapply(x -> abs.(x),
                    interp_u - interp_analytic)))
                sol.errors[:L2] = norm(sqrt(recursive_mean(vecvecapply(x -> float.(x) .^ 2,
                    interp_u .-
                    interp_analytic))))
            end
        end
    end

    nothing
end

function build_solution(sol::AbstractDAESolution{T, N}, u_analytic, errors) where {T, N}
    @reset sol.u_analytic = u_analytic
    return @set sol.errors = errors
end

function solution_new_retcode(sol::AbstractDAESolution{T, N}, retcode) where {T, N}
    return @set sol.retcode = retcode
end

function solution_new_tslocation(sol::AbstractDAESolution{T, N}, tslocation) where {T, N}
    return @set sol.tslocation = tslocation
end

function solution_slice(sol::AbstractDAESolution{T, N}, I) where {T, N}
    @reset sol.u = sol.u[I]
    @reset sol.du = sol.du[I]
    @reset sol.u_analytic = sol.u_analytic === nothing ? nothing : sol.u_analytic[I]
    @reset sol.t = sol.t[I]
    @reset sol.k = sol.dense ? sol.k[I] : sol.k
    return @set sol.dense = false
end
