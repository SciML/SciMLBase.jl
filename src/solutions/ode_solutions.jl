"""
$(TYPEDEF)

Statistics from the differential equation solver about the solution process.

## Fields

- nf: Number of function evaluations. If the differential equation is a split function,
  such as a `SplitFunction` for implicit-explicit (IMEX) integration, then `nf` is the
  number of function evaluations for the first function (the implicit function)
- nf2: If the differential equation is a split function, such as a `SplitFunction`
  for implicit-explicit (IMEX) integration, then `nf2` is the number of function
  evaluations for the second function, i.e. the function treated explicitly. Otherwise
  it is zero.
- nw: The number of W=I-gamma*J (or W=I/gamma-J) matrices constructed during the solving
  process.
- nsolve: The number of linear solves `W\b` required for the integration.
- njacs: Number of Jacobians calculated during the integration.
- nnonliniter: Total number of iterations for the nonlinear solvers.
- nnonlinconvfail: Number of nonlinear solver convergence failures.
- ncondition: Number of calls to the condition function for callbacks.
- naccept: Number of accepted steps.
- nreject: Number of rejected steps.
- maxeig: Maximum eigenvalue over the solution. This is only computed if the
  method is an auto-switching algorithm.
"""
mutable struct DEStats
    nf::Int
    nf2::Int
    nw::Int
    nsolve::Int
    njacs::Int
    nnonliniter::Int
    nnonlinconvfail::Int
    nfpiter::Int
    nfpconvfail::Int
    ncondition::Int
    naccept::Int
    nreject::Int
    maxeig::Float64
end

DEStats(x::Int = -1) = DEStats(x, x, x, x, x, x, x, x, x, x, x, x, 0.0)

function Base.show(io::IO, ::MIME"text/plain", s::DEStats)
    println(io, summary(s))
    @printf io "%-50s %-d\n" "Number of function 1 evaluations:" s.nf
    @printf io "%-50s %-d\n" "Number of function 2 evaluations:" s.nf2
    @printf io "%-50s %-d\n" "Number of W matrix evaluations:" s.nw
    @printf io "%-50s %-d\n" "Number of linear solves:" s.nsolve
    @printf io "%-50s %-d\n" "Number of Jacobians created:" s.njacs
    @printf io "%-50s %-d\n" "Number of nonlinear solver iterations:" s.nnonliniter
    @printf io "%-50s %-d\n" "Number of nonlinear solver convergence failures:" s.nnonlinconvfail
    @printf io "%-60s %-d\n" "Number of fixed-point solver iterations:" s.nfpiter
    @printf io "%-60s %-d\n" "Number of fixed-point solver convergence failures:" s.nfpconvfail
    @printf io "%-50s %-d\n" "Number of rootfind condition calls:" s.ncondition
    @printf io "%-50s %-d\n" "Number of accepted steps:" s.naccept
    @printf io "%-50s %-d" "Number of rejected steps:" s.nreject
    iszero(s.maxeig) || @printf io "\n%-50s %-d" "Maximum eigenvalue recorded:" s.maxeig
end

function Base.merge(a::DEStats, b::DEStats)
    DEStats(
        a.nf + b.nf,
        a.nf2 + b.nf2,
        a.nw + b.nw,
        a.nsolve + b.nsolve,
        a.njacs + b.njacs,
        a.nnonliniter + b.nnonliniter,
        a.nnonlinconvfail + b.nnonlinconvfail,
        a.nfpiter + b.nfpiter,
        a.nfpconvfail + b.nfpconvfail,
        a.ncondition + b.ncondition,
        a.naccept + b.naccept,
        a.nreject + b.nreject,
        max(a.maxeig, b.maxeig)
    )
end

"""
$(TYPEDEF)

Representation of the solution to an ordinary differential equation defined by an ODEProblem.

## DESolution Interface

For more information on interacting with `DESolution` types, check out the Solution Handling
page of the DifferentialEquations.jl documentation.

https://docs.sciml.ai/DiffEqDocs/stable/basics/solution/

## Fields

- `u`: the representation of the ODE solution. Given as an array of solutions, where `u[i]`
  corresponds to the solution at time `t[i]`. It is recommended in most cases one does not
  access `sol.u` directly and instead use the array interface described in the Solution
  Handling page of the DifferentialEquations.jl documentation.
- `t`: the time points corresponding to the saved values of the ODE solution.
- `prob`: the original ODEProblem that was solved.
- `alg`: the algorithm type used by the solver.
- `stats`: statistics of the solver, such as the number of function evaluations required,
  number of Jacobians computed, and more.
- `retcode`: the return code from the solver. Used to determine whether the solver solved
  successfully, whether it terminated early due to a user-defined callback, or whether it
  exited due to an error. For more details, see
  [the return code documentation](https://docs.sciml.ai/SciMLBase/stable/interfaces/Solutions/#retcodes).
"""
struct ODESolution{T, N, uType, uType2, DType, tType, rateType, P, A, IType, S,
    AC <: Union{Nothing, Vector{Int}}} <:
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
    stats::S
    alg_choice::AC
    retcode::ReturnCode.T
end

Base.@propagate_inbounds function Base.getproperty(x::AbstractODESolution, s::Symbol)
    if s === :destats
        Base.depwarn("`sol.destats` is deprecated. Use `sol.stats` instead.", "sol.destats")
        return getfield(x, :stats)
    elseif s === :ps
        return ParameterIndexingProxy(x)
    end
    return getfield(x, s)
end

function ODESolution{T, N}(u, u_analytic, errors, t, k, prob, alg, interp, dense,
        tslocation, stats, alg_choice, retcode) where {T, N}
    return ODESolution{T, N, typeof(u), typeof(u_analytic), typeof(errors), typeof(t),
        typeof(k), typeof(prob), typeof(alg), typeof(interp),
        typeof(stats),
        typeof(alg_choice)}(u, u_analytic, errors, t, k, prob, alg, interp,
        dense, tslocation, stats, alg_choice, retcode)
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
    p = hasproperty(sol.prob, :p) ? sol.prob.p : nothing
    return DiffEqArray(A.u, A.t, p, sol)
end
function (sol::AbstractODESolution)(t::AbstractVector{<:Number}, ::Type{deriv},
        idxs::AbstractVector{<:Integer},
        continuity) where {deriv}
    A = sol.interp(t, idxs, deriv, sol.prob.p, continuity)
    p = hasproperty(sol.prob, :p) ? sol.prob.p : nothing
    return DiffEqArray(A.u, A.t, p, sol)
end

function (sol::AbstractODESolution)(t::Number, ::Type{deriv}, idxs,
        continuity) where {deriv}
    symbolic_type(idxs) == NotSymbolic() && error("Incorrect specification of `idxs`")
    if is_parameter(sol, idxs)
        return getp(sol, idxs)(sol)
    else
        return augment(sol.interp([t], nothing, deriv, sol.prob.p, continuity), sol)[idxs][1]
    end
end

function (sol::AbstractODESolution)(t::Number, ::Type{deriv}, idxs::AbstractVector,
        continuity) where {deriv}
    all(!isequal(NotSymbolic()), symbolic_type.(idxs)) ||
        error("Incorrect specification of `idxs`")
    interp_sol = augment(sol.interp([t], nothing, deriv, sol.prob.p, continuity), sol)
    [is_parameter(sol, idx) ? getp(sol, idx)(sol) : first(interp_sol[idx]) for idx in idxs]
end

function (sol::AbstractODESolution)(t::AbstractVector{<:Number}, ::Type{deriv}, idxs,
        continuity) where {deriv}
    symbolic_type(idxs) == NotSymbolic() && error("Incorrect specification of `idxs`")
    if is_parameter(sol, idxs)
        return getp(sol, idxs)(sol)
    else
        interp_sol = augment(sol.interp(t, nothing, deriv, sol.prob.p, continuity), sol)
        p = hasproperty(sol.prob, :p) ? sol.prob.p : nothing
        return DiffEqArray(interp_sol[idxs], t, p, sol)
    end
end

function (sol::AbstractODESolution)(t::AbstractVector{<:Number}, ::Type{deriv},
        idxs::AbstractVector, continuity) where {deriv}
    all(!isequal(NotSymbolic()), symbolic_type.(idxs)) ||
        error("Incorrect specification of `idxs`")
    interp_sol = augment(sol.interp(t, nothing, deriv, sol.prob.p, continuity), sol)
    p = hasproperty(sol.prob, :p) ? sol.prob.p : nothing
    return DiffEqArray(
        [[interp_sol[idx][i] for idx in idxs] for i in 1:length(t)], t, p, sol)
end

function build_solution(prob::Union{AbstractODEProblem, AbstractDDEProblem},
        alg, t, u; timeseries_errors = length(u) > 2,
        dense = false, dense_errors = dense,
        calculate_error = true,
        k = nothing,
        alg_choice = nothing,
        interp = LinearInterpolation(t, u),
        retcode = ReturnCode.Default, destats = missing, stats = nothing,
        kwargs...)
    T = eltype(eltype(u))

    if prob.u0 === nothing
        N = 2
    else
        N = length((size(prob.u0)..., length(u)))
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
        sol = ODESolution{T, N}(u,
            u_analytic,
            errors,
            t, k,
            prob,
            alg,
            interp,
            dense,
            0,
            stats,
            alg_choice,
            retcode)
        if calculate_error
            calculate_solution_errors!(sol; timeseries_errors = timeseries_errors,
                dense_errors = dense_errors)
        end
        return sol
    else
        return ODESolution{T, N}(u,
            nothing,
            nothing,
            t, k,
            prob,
            alg,
            interp,
            dense,
            0,
            stats,
            alg_choice,
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
                sol.errors[:L2] = norm(sqrt(recursive_mean(vecvecapply(
                    (x) -> float.(x) .^ 2,
                    interp_u -
                    interp_analytic))))
            end
        end
    end
end

function build_solution(sol::ODESolution{T, N}, u_analytic, errors) where {T, N}
    ODESolution{T, N}(sol.u,
        u_analytic,
        errors,
        sol.t,
        sol.k,
        sol.prob,
        sol.alg,
        sol.interp,
        sol.dense,
        sol.tslocation,
        sol.stats,
        sol.alg_choice,
        sol.retcode)
end

function solution_new_retcode(sol::ODESolution{T, N}, retcode) where {T, N}
    ODESolution{T, N}(sol.u,
        sol.u_analytic,
        sol.errors,
        sol.t,
        sol.k,
        sol.prob,
        sol.alg,
        sol.interp,
        sol.dense,
        sol.tslocation,
        sol.stats,
        sol.alg_choice,
        retcode)
end

function solution_new_tslocation(sol::ODESolution{T, N}, tslocation) where {T, N}
    ODESolution{T, N}(sol.u,
        sol.u_analytic,
        sol.errors,
        sol.t,
        sol.k,
        sol.prob,
        sol.alg,
        sol.interp,
        sol.dense,
        tslocation,
        sol.stats,
        sol.alg_choice,
        sol.retcode)
end

function solution_slice(sol::ODESolution{T, N}, I) where {T, N}
    ODESolution{T, N}(sol.u[I],
        sol.u_analytic === nothing ? nothing : sol.u_analytic[I],
        sol.errors,
        sol.t[I],
        sol.dense ? sol.k[I] : sol.k,
        sol.prob,
        sol.alg,
        sol.interp,
        false,
        sol.tslocation,
        sol.stats,
        sol.alg_choice,
        sol.retcode)
end

function sensitivity_solution(sol::ODESolution, u, t)
    T = eltype(eltype(u))

    # handle save_idxs
    u0 = first(u)
    if u0 isa Number
        N = 1
    else
        N = length((size(u0)..., length(u)))
    end

    interp = enable_interpolation_sensitivitymode(sol.interp)
    ODESolution{T, N}(u, sol.u_analytic, sol.errors,
        t isa Vector ? t : collect(t),
        sol.k, sol.prob,
        sol.alg, interp,
        sol.dense, sol.tslocation,
        sol.stats, sol.alg_choice, sol.retcode)
end
