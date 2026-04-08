### Concrete Types

"""
RODESolution is an alias for ODESolution. RODE/SDE solutions are represented using the
unified ODESolution type, with the `W` field storing the noise process and `seed` storing
the random seed.
"""
const RODESolution = ODESolution

function build_solution(
        prob::Union{AbstractRODEProblem, AbstractSDDEProblem},
        alg, t, u; W = nothing, timeseries_errors = length(u) > 2,
        dense = false, dense_errors = dense, calculate_error = true,
        interp = LinearInterpolation(t, u),
        retcode = ReturnCode.Default,
        alg_choice = nothing,
        seed = UInt64(0), destats = missing, stats = nothing,
        saved_subsystem = nothing, kwargs...
    )
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

    ps = parameter_values(prob)
    if has_sys(prob.f)
        sswf = if saved_subsystem === nothing
            prob.f.sys
        else
            SavedSubsystemWithFallback(saved_subsystem, prob.f.sys)
        end
        discretes = create_parameter_timeseries_collection(sswf, ps, prob.tspan)
    else
        discretes = nothing
    end
    if has_analytic(f)
        u_analytic = Vector{typeof(prob.u0)}()
        errors = Dict{Symbol, real(eltype(prob.u0))}()
        sol = ODESolution{T, N}(
            u,
            nothing,
            u_analytic,
            errors,
            t, nothing,
            W,
            discretes,
            prob,
            alg,
            interp,
            dense,
            0,
            stats,
            alg_choice,
            retcode,
            nothing,
            nothing,
            seed,
            saved_subsystem
        )

        if calculate_error
            calculate_solution_errors!(
                sol; timeseries_errors = timeseries_errors,
                dense_errors = dense_errors
            )
        end

        return sol
    else
        return ODESolution{T, N}(
            u,
            nothing,
            nothing,
            nothing,
            t, nothing,
            W,
            discretes,
            prob,
            alg,
            interp,
            dense,
            0,
            stats,
            alg_choice,
            retcode,
            nothing,
            nothing,
            seed,
            saved_subsystem
        )
    end
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
