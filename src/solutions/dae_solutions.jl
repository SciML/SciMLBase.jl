"""
DAESolution is an alias for ODESolution. DAE solutions are represented using the unified
ODESolution type, with the `du` field storing the derivative values.
"""
const DAESolution = ODESolution

function build_solution(
        prob::AbstractDAEProblem, alg, t, u, du = nothing;
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
        kwargs...
    )
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
    return if has_analytic(prob.f)
        u_analytic = Vector{typeof(prob.u0)}()
        errors = Dict{Symbol, real(eltype(prob.u0))}()

        sol = ODESolution{T, N}(
            u,
            du,
            u_analytic,
            errors,
            t,
            k,
            nothing,
            discretes,
            prob,
            alg,
            interp,
            dense,
            0,
            stats,
            nothing,
            retcode,
            nothing,
            nothing,
            UInt64(0),
            saved_subsystem
        )

        if calculate_error
            calculate_solution_errors!(
                sol; timeseries_errors = timeseries_errors,
                dense_errors = dense_errors
            )
        end
        sol
    else
        ODESolution{T, N}(
            u,
            du,
            nothing,
            nothing,
            t,
            k,
            nothing,
            discretes,
            prob,
            alg,
            interp,
            dense,
            0,
            stats,
            nothing,
            retcode,
            nothing,
            nothing,
            UInt64(0),
            saved_subsystem
        )
    end
end
