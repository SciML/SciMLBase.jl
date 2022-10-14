"""
$(TYPEDEF)

Representation of the solution to an differential-algebraic equation defined by an DAEProblem.

## DESolution Interface

For more information on interacting with `DESolution` types, check out the Solution Handling
page of the DifferentialEquations.jl documentation.

https://diffeq.sciml.ai/stable/basics/solution/

## Fields

- `u`: the representation of the DAE solution. Given as an array of solutions, where `u[i]`
  corresponds to the solution at time `t[i]`. It is recommended in most cases one does not
  access `sol.u` directly and instead use the array interface described in the Solution
  Handling page of the DifferentialEquations.jl documentation.
- `du`: the representation of the derivatives of the DAE solution.
- `t`: the time points corresponding to the saved values of the DAE solution.
- `prob`: the original DAEProblem that was solved.
- `alg`: the algorithm type used by the solver.
- `destats`: statistics of the solver, such as the number of function evaluations required,
  number of Jacobians computed, and more.
- `retcode`: the return code from the solver. Used to determine whether the solver solved
  successfully (`sol.retcode === ReturnCode.Success`), whether it terminated due to a user-defined
  callback (`sol.retcode === ReturnCode.Terminated`), or whether it exited due to an error. For more
  details, see the return code section of the DifferentialEquations.jl documentation.
"""
struct DAESolution{T, N, uType, duType, uType2, DType, tType, P, A, ID, DE} <:
       AbstractDAESolution{T, N, uType}
    u::uType
    du::duType
    u_analytic::uType2
    errors::DType
    t::tType
    prob::P
    alg::A
    interp::ID
    dense::Bool
    tslocation::Int
    destats::DE
    retcode::ReturnCode.T
end
function (sol::DAESolution)(t, ::Type{deriv} = Val{0}; idxs = nothing,
                            continuity = :left) where {deriv}
    sol.interp(t, idxs, deriv, sol.prob.p, continuity)
end
function (sol::DAESolution)(v, t, ::Type{deriv} = Val{0}; idxs = nothing,
                            continuity = :left) where {deriv}
    sol.interp(v, t, idxs, deriv, sol.prob.p, continuity)
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
                        destats = nothing,
                        kwargs...)
    T = eltype(eltype(u))
    N = length((size(prob.u0)..., length(u)))

    if has_analytic(prob.f)
        u_analytic = Vector{typeof(prob.u0)}()
        errors = Dict{Symbol, real(eltype(prob.u0))}()

        sol = DAESolution{T, N, typeof(u), typeof(du), typeof(u_analytic), typeof(errors),
                          typeof(t),
                          typeof(prob), typeof(alg), typeof(interp), typeof(destats)}(u, du,
                                                                                      u_analytic,
                                                                                      errors,
                                                                                      t,
                                                                                      prob,
                                                                                      alg,
                                                                                      interp,
                                                                                      dense,
                                                                                      0,
                                                                                      destats,
                                                                                      retcode)

        if calculate_error
            calculate_solution_errors!(sol; timeseries_errors = timeseries_errors,
                                       dense_errors = dense_errors)
        end
        sol
    else
        DAESolution{T, N, typeof(u), typeof(du), Nothing, Nothing, typeof(t),
                    typeof(prob), typeof(alg), typeof(interp), typeof(destats)}(u, du,
                                                                                nothing,
                                                                                nothing, t,
                                                                                prob, alg,
                                                                                interp,
                                                                                dense, 0,
                                                                                destats,
                                                                                retcode)
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
    DAESolution{T, N, typeof(sol.u), typeof(sol.du), typeof(u_analytic), typeof(errors),
                typeof(sol.t),
                typeof(sol.prob), typeof(sol.alg), typeof(sol.interp), typeof(sol.destats)}(sol.u,
                                                                                            sol.du,
                                                                                            u_analytic,
                                                                                            errors,
                                                                                            sol.t,
                                                                                            sol.prob,
                                                                                            sol.alg,
                                                                                            sol.interp,
                                                                                            sol.dense,
                                                                                            sol.tslocation,
                                                                                            sol.destats,
                                                                                            sol.retcode)
end

function solution_new_retcode(sol::AbstractDAESolution{T, N}, retcode) where {T, N}
    DAESolution{T, N, typeof(sol.u), typeof(sol.du), typeof(sol.u_analytic),
                typeof(sol.errors), typeof(sol.t),
                typeof(sol.prob), typeof(sol.alg), typeof(sol.interp), typeof(sol.destats)}(sol.u,
                                                                                            sol.du,
                                                                                            sol.u_analytic,
                                                                                            sol.errors,
                                                                                            sol.t,
                                                                                            sol.prob,
                                                                                            sol.alg,
                                                                                            sol.interp,
                                                                                            sol.dense,
                                                                                            sol.tslocation,
                                                                                            sol.destats,
                                                                                            retcode)
end

function solution_new_tslocation(sol::AbstractDAESolution{T, N}, tslocation) where {T, N}
    DAESolution{T, N, typeof(sol.u), typeof(sol.du), typeof(sol.u_analytic),
                typeof(sol.errors), typeof(sol.t),
                typeof(sol.prob), typeof(sol.alg), typeof(sol.interp), typeof(sol.destats)}(sol.u,
                                                                                            sol.du,
                                                                                            sol.u_analytic,
                                                                                            sol.errors,
                                                                                            sol.t,
                                                                                            sol.prob,
                                                                                            sol.alg,
                                                                                            sol.interp,
                                                                                            sol.dense,
                                                                                            tslocation,
                                                                                            sol.destats,
                                                                                            sol.retcode)
end

function solution_slice(sol::AbstractDAESolution{T, N}, I) where {T, N}
    DAESolution{T, N, typeof(sol.u), typeof(sol.du), typeof(sol.u_analytic),
                typeof(sol.errors), typeof(sol.t),
                typeof(sol.prob), typeof(sol.alg), typeof(sol.interp), typeof(sol.destats)}(sol.u[I],
                                                                                            sol.du[I],
                                                                                            sol.u_analytic ===
                                                                                            nothing ?
                                                                                            nothing :
                                                                                            sol.u_analytic[I],
                                                                                            sol.errors,
                                                                                            sol.t[I],
                                                                                            sol.prob,
                                                                                            sol.alg,
                                                                                            sol.interp,
                                                                                            false,
                                                                                            sol.tslocation,
                                                                                            sol.destats,
                                                                                            sol.retcode)
end
