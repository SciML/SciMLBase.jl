"""
$(TYPEDEF)

Representation of the solution to an ordinary differential equation defined by an ODEProblem.

## DESolution Interface

For more information on interacting with `DESolution` types, check out the Solution Handling
page of the DifferentialEquations.jl documentation.

https://diffeq.sciml.ai/stable/basics/solution/

## Fields

- `u`: the representation of the ODE solution. Given as an array of solutions, where `u[i]`
  corresponds to the solution at time `t[i]`. It is recommended in most cases one does not
  access `sol.u` directly and instead use the array interface described in the Solution 
  Handling page of the DifferentialEquations.jl documentation.
- `t`: the time points corresponding to the saved values of the ODE solution.
- `prob`: the original ODEProblem that was solved.
- `alg`: the algorithm type used by the solver.
- `destats`: statistics of the solver, such as the number of function evaluations required,
  number of Jacobians computed, and more.
- `retcode`: the return code from the solver. Used to determine whether the solver solved
  successfully (`sol.retcode === :Success`), whether it terminated due to a user-defined
  callback (`sol.retcode === :Terminated`), or whether it exited due to an error. For more
  details, see the return code section of the DifferentialEquations.jl documentation.
"""
struct ODESolution{T, N, uType, uType2, DType, tType, rateType, P, A, IType, DE} <:
       AbstractODESolution{T, N, uType}
    u::uType
    u_analytic::uType2
    errors::DType
    t::tType
    k::rateType
    prob::P
    alg::A
    interp::IType
    dense::Bool
    tslocation::Int
    destats::DE
    retcode::Symbol
end
function (sol::AbstractODESolution)(t, ::Type{deriv} = Val{0}; idxs = nothing,
                                    continuity = :left) where {deriv}
    sol(t, deriv, idxs, continuity)
end
function (sol::AbstractODESolution)(v, t, ::Type{deriv} = Val{0}; idxs = nothing,
                                    continuity = :left) where {deriv}
    sol.interp(v, t, idxs, deriv, sol.prob.p, continuity)
end

function (sol::AbstractODESolution)(t::Number, ::Type{deriv}, idxs::Nothing,
                                    continuity) where {deriv}
    sol.interp(t, idxs, deriv, sol.prob.p, continuity)
end

function (sol::AbstractODESolution)(t::AbstractVector{<:Number}, ::Type{deriv},
                                    idxs::Nothing, continuity) where {deriv}
    augment(sol.interp(t, idxs, deriv, sol.prob.p, continuity), sol)
end

function (sol::AbstractODESolution)(t::Number, ::Type{deriv}, idxs::Integer,
                                    continuity) where {deriv}
    sol.interp(t, idxs, deriv, sol.prob.p, continuity)
end
function (sol::AbstractODESolution)(t::Number, ::Type{deriv},
                                    idxs::AbstractVector{<:Integer},
                                    continuity) where {deriv}
    sol.interp(t, idxs, deriv, sol.prob.p, continuity)
end
function (sol::AbstractODESolution)(t::AbstractVector{<:Number}, ::Type{deriv},
                                    idxs::Integer, continuity) where {deriv}
    A = sol.interp(t, idxs, deriv, sol.prob.p, continuity)
    syms = hasproperty(sol.prob.f, :syms) && sol.prob.f.syms !== nothing ?
           [sol.prob.f.syms[idxs]] : nothing
    observed = has_observed(sol.prob.f) ? sol.prob.f.observed : DEFAULT_OBSERVED
    p = hasproperty(sol.prob, :p) ? sol.prob.p : nothing
    DiffEqArray(A.u, A.t, syms, getindepsym(sol), observed, p)
end
function (sol::AbstractODESolution)(t::AbstractVector{<:Number}, ::Type{deriv},
                                    idxs::AbstractVector{<:Integer},
                                    continuity) where {deriv}
    A = sol.interp(t, idxs, deriv, sol.prob.p, continuity)
    syms = hasproperty(sol.prob.f, :syms) && sol.prob.f.syms !== nothing ?
           sol.prob.f.syms[idxs] : nothing
    observed = has_observed(sol.prob.f) ? sol.prob.f.observed : DEFAULT_OBSERVED
    p = hasproperty(sol.prob, :p) ? sol.prob.p : nothing
    DiffEqArray(A.u, A.t, syms, getindepsym(sol), observed, p)
end

function (sol::AbstractODESolution)(t::Number, ::Type{deriv}, idxs,
                                    continuity) where {deriv}
    issymbollike(idxs) || error("Incorrect specification of `idxs`")
    augment(sol.interp([t], nothing, deriv, sol.prob.p, continuity), sol)[idxs][1]
end

function (sol::AbstractODESolution)(t::Number, ::Type{deriv}, idxs::AbstractVector,
                                    continuity) where {deriv}
    all(issymbollike.(idxs)) || error("Incorrect specification of `idxs`")
    interp_sol = augment(sol.interp([t], nothing, deriv, sol.prob.p, continuity), sol)
    [first(interp_sol[idx]) for idx in idxs]
end

function (sol::AbstractODESolution)(t::AbstractVector{<:Number}, ::Type{deriv}, idxs,
                                    continuity) where {deriv}
    issymbollike(idxs) || error("Incorrect specification of `idxs`")
    interp_sol = augment(sol.interp(t, nothing, deriv, sol.prob.p, continuity), sol)
    observed = has_observed(sol.prob.f) ? sol.prob.f.observed : DEFAULT_OBSERVED
    p = hasproperty(sol.prob, :p) ? sol.prob.p : nothing
    DiffEqArray(interp_sol[idxs], t, [idxs], getindepsym(sol), observed, p)
end

function (sol::AbstractODESolution)(t::AbstractVector{<:Number}, ::Type{deriv},
                                    idxs::AbstractVector, continuity) where {deriv}
    all(issymbollike.(idxs)) || error("Incorrect specification of `idxs`")
    interp_sol = augment(sol.interp(t, nothing, deriv, sol.prob.p, continuity), sol)
    observed = has_observed(sol.prob.f) ? sol.prob.f.observed : DEFAULT_OBSERVED
    p = hasproperty(sol.prob, :p) ? sol.prob.p : nothing
    DiffEqArray([[interp_sol[idx][i] for idx in idxs] for i in 1:length(t)], t, idxs,
                getindepsym(sol), observed, p)
end

function build_solution(prob::Union{AbstractODEProblem, AbstractDDEProblem},
                        alg, t, u; timeseries_errors = length(u) > 2,
                        dense = false, dense_errors = dense,
                        calculate_error = true,
                        k = nothing,
                        interp = LinearInterpolation(t, u),
                        retcode = :Default, destats = nothing, kwargs...)
    T = eltype(eltype(u))
    N = length((size(prob.u0)..., length(u)))

    if typeof(prob.f) <: Tuple
        f = prob.f[1]
    else
        f = prob.f
    end

    if has_analytic(f)
        u_analytic = Vector{typeof(prob.u0)}()
        errors = Dict{Symbol, real(eltype(prob.u0))}()
        sol = ODESolution{T, N, typeof(u), typeof(u_analytic), typeof(errors), typeof(t),
                          typeof(k),
                          typeof(prob), typeof(alg), typeof(interp), typeof(destats)}(u,
                                                                                      u_analytic,
                                                                                      errors,
                                                                                      t, k,
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
        return sol
    else
        return ODESolution{T, N, typeof(u), Nothing, Nothing, typeof(t), typeof(k),
                           typeof(prob), typeof(alg), typeof(interp), typeof(destats)}(u,
                                                                                       nothing,
                                                                                       nothing,
                                                                                       t, k,
                                                                                       prob,
                                                                                       alg,
                                                                                       interp,
                                                                                       dense,
                                                                                       0,
                                                                                       destats,
                                                                                       retcode)
    end
end

function calculate_solution_errors!(sol::AbstractODESolution; fill_uanalytic = true,
                                    timeseries_errors = true, dense_errors = true)
    f = sol.prob.f

    if fill_uanalytic
        for i in 1:size(sol.u, 1)
            if sol.prob isa AbstractDDEProblem
                push!(sol.u_analytic,
                      f.analytic(sol.prob.u0, sol.prob.h, sol.prob.p, sol.t[i]))
            else
                push!(sol.u_analytic, f.analytic(sol.prob.u0, sol.prob.p, sol.t[i]))
            end
        end
    end

    save_everystep = length(sol.u) > 2
    if !isempty(sol.u_analytic)
        sol.errors[:final] = norm(recursive_mean(abs.(sol.u[end] .- sol.u_analytic[end])))

        if save_everystep && timeseries_errors
            sol.errors[:l∞] = norm(maximum(vecvecapply((x) -> abs.(x),
                                                       sol.u - sol.u_analytic)))
            sol.errors[:l2] = norm(sqrt(recursive_mean(vecvecapply((x) -> float.(x) .^ 2,
                                                                   sol.u - sol.u_analytic))))
            if sol.dense && dense_errors
                densetimes = collect(range(sol.t[1], stop = sol.t[end], length = 100))
                interp_u = sol(densetimes)
                interp_analytic = VectorOfArray([f.analytic(sol.prob.u0, sol.prob.p, t)
                                                 for t in densetimes])
                sol.errors[:L∞] = norm(maximum(vecvecapply((x) -> abs.(x),
                                                           interp_u - interp_analytic)))
                sol.errors[:L2] = norm(sqrt(recursive_mean(vecvecapply((x) -> float.(x) .^ 2,
                                                                       interp_u -
                                                                       interp_analytic))))
            end
        end
    end
end

function build_solution(sol::AbstractODESolution{T, N}, u_analytic, errors) where {T, N}
    ODESolution{T, N, typeof(sol.u), typeof(u_analytic), typeof(errors), typeof(sol.t),
                typeof(sol.k),
                typeof(sol.prob), typeof(sol.alg), typeof(sol.interp), typeof(sol.destats)}(sol.u,
                                                                                            u_analytic,
                                                                                            errors,
                                                                                            sol.t,
                                                                                            sol.k,
                                                                                            sol.prob,
                                                                                            sol.alg,
                                                                                            sol.interp,
                                                                                            sol.dense,
                                                                                            sol.tslocation,
                                                                                            sol.destats,
                                                                                            sol.retcode)
end

function solution_new_retcode(sol::AbstractODESolution{T, N}, retcode) where {T, N}
    ODESolution{T, N, typeof(sol.u), typeof(sol.u_analytic), typeof(sol.errors),
                typeof(sol.t), typeof(sol.k),
                typeof(sol.prob), typeof(sol.alg), typeof(sol.interp), typeof(sol.destats)}(sol.u,
                                                                                            sol.u_analytic,
                                                                                            sol.errors,
                                                                                            sol.t,
                                                                                            sol.k,
                                                                                            sol.prob,
                                                                                            sol.alg,
                                                                                            sol.interp,
                                                                                            sol.dense,
                                                                                            sol.tslocation,
                                                                                            sol.destats,
                                                                                            retcode)
end

function solution_new_tslocation(sol::AbstractODESolution{T, N}, tslocation) where {T, N}
    ODESolution{T, N, typeof(sol.u), typeof(sol.u_analytic), typeof(sol.errors),
                typeof(sol.t), typeof(sol.k),
                typeof(sol.prob), typeof(sol.alg), typeof(sol.interp), typeof(sol.destats)}(sol.u,
                                                                                            sol.u_analytic,
                                                                                            sol.errors,
                                                                                            sol.t,
                                                                                            sol.k,
                                                                                            sol.prob,
                                                                                            sol.alg,
                                                                                            sol.interp,
                                                                                            sol.dense,
                                                                                            tslocation,
                                                                                            sol.destats,
                                                                                            sol.retcode)
end

function solution_slice(sol::AbstractODESolution{T, N}, I) where {T, N}
    ODESolution{T, N, typeof(sol.u), typeof(sol.u_analytic), typeof(sol.errors),
                typeof(sol.t), typeof(sol.k),
                typeof(sol.prob), typeof(sol.alg), typeof(sol.interp), typeof(sol.destats)}(sol.u[I],
                                                                                            sol.u_analytic ===
                                                                                            nothing ?
                                                                                            nothing :
                                                                                            sol.u_analytic[I],
                                                                                            sol.errors,
                                                                                            sol.t[I],
                                                                                            sol.dense ?
                                                                                            sol.k[I] :
                                                                                            sol.k,
                                                                                            sol.prob,
                                                                                            sol.alg,
                                                                                            sol.interp,
                                                                                            false,
                                                                                            sol.tslocation,
                                                                                            sol.destats,
                                                                                            sol.retcode)
end

function sensitivity_solution(sol::AbstractODESolution, u, t)
    T = eltype(eltype(u))
    N = length((size(sol.prob.u0)..., length(u)))
    interp = if typeof(sol.interp) <: LinearInterpolation
        LinearInterpolation(t, u)
    elseif typeof(sol.interp) <: ConstantInterpolation
        ConstantInterpolation(t, u)
    else
        SensitivityInterpolation(t, u)
    end

    ODESolution{T, N, typeof(u), typeof(sol.u_analytic), typeof(sol.errors),
                typeof(t), Nothing, typeof(sol.prob), typeof(sol.alg),
                typeof(interp), typeof(sol.destats)}(u, sol.u_analytic, sol.errors, t,
                                                     nothing, sol.prob,
                                                     sol.alg, interp,
                                                     sol.dense, sol.tslocation,
                                                     sol.destats, sol.retcode)
end
