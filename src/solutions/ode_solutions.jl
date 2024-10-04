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
    iszero(s.maxeig) || @printf io "\n%-50s %-e" "Maximum eigenvalue recorded:" s.maxeig
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
- `saved_subsystem`: a [`SavedSubsystem`](@ref) representing the subset of variables saved
  in this solution, or `nothing` if all variables are saved. Here "variables" refers to
  both continuous-time state variables and timeseries parameters.
"""
struct ODESolution{T, N, uType, uType2, DType, tType, rateType, discType, P, A, IType, S,
    AC <: Union{Nothing, Vector{Int}}, R, O, V} <:
       AbstractODESolution{T, N, uType}
    u::uType
    u_analytic::uType2
    errors::DType
    t::tType
    k::rateType
    discretes::discType
    prob::P
    alg::A
    interp::IType
    dense::Bool
    tslocation::Int
    stats::S
    alg_choice::AC
    retcode::ReturnCode.T
    resid::R
    original::O
    saved_subsystem::V
end

function ConstructionBase.constructorof(::Type{O}) where {T, N, O <: ODESolution{T, N}}
    ODESolution{T, N}
end

function ConstructionBase.setproperties(sol::ODESolution, patch::NamedTuple)
    u = get(patch, :u, sol.u)
    N = u === nothing ? 2 : ndims(eltype(u)) + 1
    T = eltype(eltype(u))
    patch = merge(getproperties(sol), patch)
    return ODESolution{T, N}(patch.u, patch.u_analytic, patch.errors, patch.t, patch.k,
        patch.discretes, patch.prob, patch.alg, patch.interp, patch.dense, patch.tslocation, patch.stats,
        patch.alg_choice, patch.retcode, patch.resid, patch.original, patch.saved_subsystem)
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

# FIXME: Remove the defaults for resid and original on a breaking release
function ODESolution{T, N}(
        u, u_analytic, errors, t, k, discretes, prob, alg, interp, dense,
        tslocation, stats, alg_choice, retcode, resid = nothing,
        original = nothing, saved_subsystem = nothing) where {T, N}
    return ODESolution{T, N, typeof(u), typeof(u_analytic), typeof(errors), typeof(t),
        typeof(k), typeof(discretes), typeof(prob), typeof(alg), typeof(interp),
        typeof(stats), typeof(alg_choice), typeof(resid), typeof(original),
        typeof(saved_subsystem)}(u, u_analytic, errors, t, k, discretes, prob, alg, interp,
        dense, tslocation, stats, alg_choice, retcode, resid, original, saved_subsystem)
end

error_if_observed_derivative(_, _, ::Type{Val{0}}) = nothing
function error_if_observed_derivative(sys, idx, ::Type)
    if symbolic_type(idx) != NotSymbolic() && is_observed(sys, idx) ||
       symbolic_type(idx) == NotSymbolic() && any(x -> is_observed(sys, x), idx)
        error("""
        Cannot interpolate derivatives of observed variables. A possible solution could be
        interpolating the symbolic expression that evaluates to the derivative of the
        observed variable or using DataInterpolations.jl.
        """)
    end
end

function SymbolicIndexingInterface.is_parameter_timeseries(::Type{S}) where {
        T1, T2, T3, T4, T5, T6, T7,
        S <: ODESolution{T1, T2, T3, T4, T5, T6, T7, <:ParameterTimeseriesCollection}}
    Timeseries()
end

function _hold_discrete(disc_u, disc_t, t::Number)
    idx = searchsortedlast(disc_t, t)
    if idx == firstindex(disc_t) - 1
        error("Cannot access discrete variable at time $t before initial save $(first(disc_t))")
    end
    return disc_u[idx]
end

function hold_discrete(disc_u, disc_t, t::Number)
    val = _hold_discrete(disc_u, disc_t, t)
    return DiffEqArray([val], [t])
end

function hold_discrete(disc_u, disc_t, t::AbstractVector{<:Number})
    return DiffEqArray(_hold_discrete.((disc_u,), (disc_t,), t), t)
end

function get_interpolated_discretes(sol::AbstractODESolution, t, deriv, continuity)
    is_parameter_timeseries(sol) == Timeseries() || return nothing

    discs::ParameterTimeseriesCollection = RecursiveArrayTools.get_discretes(sol)
    interp_discs = map(discs) do partition
        hold_discrete(partition.u, partition.t, t)
    end
    return ParameterTimeseriesCollection(interp_discs, parameter_values(discs))
end

function is_discrete_expression(indp, expr)
    ts_idxs = get_all_timeseries_indexes(indp, expr)
    length(ts_idxs) > 1 || length(ts_idxs) == 1 && only(ts_idxs) != ContinuousTimeseries()
end

function (sol::AbstractODESolution)(t, ::Type{deriv} = Val{0}; idxs = nothing,
        continuity = :left) where {deriv}
    if t isa IndexedClock
        t = canonicalize_indexed_clock(t, sol)
    end
    sol(t, deriv, idxs, continuity)
end
function (sol::AbstractODESolution)(v, t, ::Type{deriv} = Val{0}; idxs = nothing,
        continuity = :left) where {deriv}
    if t isa IndexedClock
        t = canonicalize_indexed_clock(t, sol)
    end
    sol.interp(v, t, idxs, deriv, sol.prob.p, continuity)
end

function (sol::AbstractODESolution)(t::Number, ::Type{deriv}, idxs::Nothing,
        continuity) where {deriv}
    sol.interp(t, idxs, deriv, sol.prob.p, continuity)
end

function (sol::AbstractODESolution)(t::AbstractVector{<:Number}, ::Type{deriv},
        idxs::Nothing, continuity) where {deriv}
    discretes = get_interpolated_discretes(sol, t, deriv, continuity)
    augment(sol.interp(t, idxs, deriv, sol.prob.p, continuity), sol; discretes)
end

function (sol::AbstractODESolution)(t::Number, ::Type{deriv}, idxs::Integer,
        continuity) where {deriv}
    sol.interp(t, idxs, deriv, sol.prob.p, continuity)
end
function (sol::AbstractODESolution)(t::Number, ::Type{deriv},
        idxs::AbstractVector{<:Integer},
        continuity) where {deriv}
    if isempty(idxs)
        return eltype(eltype(sol.u))[]
    end
    if eltype(sol.u) <: Number
        idxs = only(idxs)
    end
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
    if isempty(idxs)
        return map(_ -> eltype(eltype(sol.u))[], t)
    end
    if eltype(sol.u) <: Number
        idxs = only(idxs)
    end
    A = sol.interp(t, idxs, deriv, sol.prob.p, continuity)
    p = hasproperty(sol.prob, :p) ? sol.prob.p : nothing
    return DiffEqArray(A.u, A.t, p, sol)
end

function (sol::AbstractODESolution)(t::Number, ::Type{deriv}, idxs,
        continuity) where {deriv}
    symbolic_type(idxs) == NotSymbolic() && error("Incorrect specification of `idxs`")
    error_if_observed_derivative(sol, idxs, deriv)
    ps = parameter_values(sol)
    if is_parameter(sol, idxs) && !is_timeseries_parameter(sol, idxs)
        return getp(sol, idxs)(ps)
    end
    if is_parameter_timeseries(sol) == Timeseries() && is_discrete_expression(sol, idxs)
        discs::ParameterTimeseriesCollection = RecursiveArrayTools.get_discretes(sol)
        ps = parameter_values(discs)
        for ts_idx in eachindex(discs)
            partition = discs[ts_idx]
            interp_val = ConstantInterpolation(partition.t, partition.u)(
                t, nothing, deriv, nothing, continuity)
            ps = with_updated_parameter_timeseries_values(sol, ps, ts_idx => interp_val)
        end
    end
    state = ProblemState(; u = sol.interp(t, nothing, deriv, ps, continuity), p = ps, t)
    return getu(sol, idxs)(state)
end

function (sol::AbstractODESolution)(t::Number, ::Type{deriv}, idxs::AbstractVector,
        continuity) where {deriv}
    if symbolic_type(idxs) == NotSymbolic() &&
       any(isequal(NotSymbolic()), symbolic_type.(idxs))
        error("Incorrect specification of `idxs`")
    end
    if symbolic_type(idxs) == NotSymbolic() && isempty(idxs)
        return eltype(eltype(sol.u))[]
    end
    error_if_observed_derivative(sol, idxs, deriv)
    ps = parameter_values(sol)
    if is_parameter_timeseries(sol) == Timeseries() && is_discrete_expression(sol, idxs)
        discs::ParameterTimeseriesCollection = RecursiveArrayTools.get_discretes(sol)
        ps = parameter_values(discs)
        for ts_idx in eachindex(discs)
            partition = discs[ts_idx]
            interp_val = ConstantInterpolation(partition.t, partition.u)(
                t, nothing, deriv, nothing, continuity)
            ps = with_updated_parameter_timeseries_values(sol, ps, ts_idx => interp_val)
        end
    end
    state = ProblemState(; u = sol.interp(t, nothing, deriv, ps, continuity), p = ps, t)
    return getu(sol, idxs)(state)
end

function (sol::AbstractODESolution)(t::AbstractVector{<:Number}, ::Type{deriv}, idxs,
        continuity) where {deriv}
    symbolic_type(idxs) == NotSymbolic() && error("Incorrect specification of `idxs`")
    error_if_observed_derivative(sol, idxs, deriv)
    p = hasproperty(sol.prob, :p) ? sol.prob.p : nothing
    getter = getu(sol, idxs)
    if is_parameter_timeseries(sol) == NotTimeseries() || !is_discrete_expression(sol, idxs)
        interp_sol = augment(sol.interp(t, nothing, deriv, p, continuity), sol)
        return DiffEqArray(getter(interp_sol), t, p, sol)
    end
    discretes = get_interpolated_discretes(sol, t, deriv, continuity)
    interp_sol = sol.interp(t, nothing, deriv, p, continuity)
    u = map(eachindex(t)) do ti
        ps = parameter_values(discretes)
        for i in eachindex(discretes)
            ps = with_updated_parameter_timeseries_values(sol, ps, i => discretes[i, ti])
        end
        return getter(ProblemState(; u = interp_sol.u[ti], p = ps, t = t[ti]))
    end
    return DiffEqArray(u, t, p, sol; discretes)
end

function (sol::AbstractODESolution)(t::AbstractVector{<:Number}, ::Type{deriv},
        idxs::AbstractVector, continuity) where {deriv}
    if symbolic_type(idxs) == NotSymbolic() && isempty(idxs)
        return map(_ -> eltype(eltype(sol.u))[], t)
    end
    error_if_observed_derivative(sol, idxs, deriv)
    p = hasproperty(sol.prob, :p) ? sol.prob.p : nothing
    getter = getu(sol, idxs)
    if is_parameter_timeseries(sol) == NotTimeseries() || !is_discrete_expression(sol, idxs)
        interp_sol = augment(sol.interp(t, nothing, deriv, p, continuity), sol)
        return DiffEqArray(getter(interp_sol), t, p, sol)
    end
    discretes = get_interpolated_discretes(sol, t, deriv, continuity)
    interp_sol = sol.interp(t, nothing, deriv, p, continuity)
    u = map(eachindex(t)) do ti
        ps = parameter_values(discretes)
        for i in eachindex(discretes)
            ps = with_updated_parameter_timeseries_values(sol, ps, i => discretes[i, ti])
        end
        return getter(ProblemState(; u = interp_sol.u[ti], p = ps, t = t[ti]))
    end
    return DiffEqArray(u, t, p, sol; discretes)
end

struct DDESolutionHistoryWrapper{T}
    sol::T
end

function (w::DDESolutionHistoryWrapper)(p, t; idxs = nothing)
    w.sol(t; idxs)
end
function (w::DDESolutionHistoryWrapper)(out, p, t; idxs = nothing)
    w.sol(out, t; idxs)
end
function (w::DDESolutionHistoryWrapper)(p, t, deriv::Type{Val{i}}; idxs = nothing) where {i}
    w.sol(t, deriv; idxs)
end
function (w::DDESolutionHistoryWrapper)(
        out, p, t, deriv::Type{Val{i}}; idxs = nothing) where {i}
    w.sol(out, t, deriv; idxs)
end

function SymbolicIndexingInterface.get_history_function(sol::ODESolution)
    DDESolutionHistoryWrapper(sol)
end

# public API, used by MTK
"""
    create_parameter_timeseries_collection(sys, ps, tspan)

Create a `SymbolicIndexingInterface.ParameterTimeseriesCollection` for the given system
`sys` and parameter object `ps`. Return `nothing` if there are no timeseries parameters.
Defaults to `nothing`. Falls back on the basis of `symbolic_container`.
"""
function create_parameter_timeseries_collection(sys, ps, tspan)
    if hasmethod(symbolic_container, Tuple{typeof(sys)})
        return create_parameter_timeseries_collection(symbolic_container(sys), ps, tspan)
    else
        return nothing
    end
end

const PeriodicDiffEqArray = DiffEqArray{T, N, A, B} where {T, N, A, B <: AbstractRange}

# public API, used by MTK
"""
    get_saveable_values(sys, ps, timeseries_idx)

Return the values to be saved in parameter object `ps` for timeseries index `timeseries_idx`. Called by
`save_discretes!`. If this returns `nothing`, `save_discretes!` will not save anything.
"""
function get_saveable_values(sys, ps, timeseries_idx)
    return get_saveable_values(symbolic_container(sys), ps, timeseries_idx)
end

"""
    save_discretes!(integ::DEIntegrator, timeseries_idx)

Save the parameter timeseries with index `timeseries_idx`. Calls `get_saveable_values` to
get the values to save. If it returns `nothing`, then the save does not happen.
"""
function save_discretes!(integ::DEIntegrator, timeseries_idx)
    inner_sol = get_sol(integ)
    vals = get_saveable_values(inner_sol, parameter_values(integ), timeseries_idx)
    vals === nothing && return
    save_discretes!(integ.sol, current_time(integ), vals, timeseries_idx)
end

save_discretes!(args...) = nothing

# public API, used by MTK
function save_discretes!(sol::AbstractODESolution, t, vals, timeseries_idx)
    RecursiveArrayTools.has_discretes(sol) || return
    disc = RecursiveArrayTools.get_discretes(sol)
    _save_discretes_internal!(disc[timeseries_idx], t, vals)
end

function _save_discretes_internal!(A::AbstractDiffEqArray, t, vals)
    push!(A.t, t)
    push!(A.u, vals)
end

function _save_discretes_internal!(A::PeriodicDiffEqArray, t, vals)
    idx = length(A.u) + 1
    if A.t[idx] ≉ t
        error("Tried to save periodic discrete value with timeseries $(A.t) at time $t")
    end
    push!(A.u, vals)
end

function build_solution(prob::Union{AbstractODEProblem, AbstractDDEProblem},
        alg, t, u; timeseries_errors = length(u) > 2,
        dense = false, dense_errors = dense,
        calculate_error = true,
        k = nothing,
        alg_choice = nothing,
        interp = LinearInterpolation(t, u),
        retcode = ReturnCode.Default, destats = missing, stats = nothing,
        resid = nothing, original = nothing,
        saved_subsystem = nothing,
        kwargs...)
    T = eltype(eltype(u))

    if prob.u0 === nothing
        N = 2
    elseif prob isa BVProblem && !hasmethod(size, Tuple{typeof(prob.u0)})
        __u0 = hasmethod(prob.u0, Tuple{typeof(prob.p), typeof(first(prob.tspan))}) ?
               prob.u0(prob.p, first(prob.tspan)) : prob.u0(first(prob.tspan))
        N = length((size(__u0)..., length(u)))
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
        sol = ODESolution{T, N}(u,
            u_analytic,
            errors,
            t, k,
            discretes,
            prob,
            alg,
            interp,
            dense,
            0,
            stats,
            alg_choice,
            retcode,
            resid,
            original,
            saved_subsystem)
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
            discretes,
            prob,
            alg,
            interp,
            dense,
            0,
            stats,
            alg_choice,
            retcode,
            resid,
            original,
            saved_subsystem)
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
    @reset sol.u_analytic = u_analytic
    return @set sol.errors = errors
end

function solution_new_retcode(sol::ODESolution{T, N}, retcode) where {T, N}
    return @set sol.retcode = retcode
end

function solution_new_tslocation(sol::ODESolution{T, N}, tslocation) where {T, N}
    return @set sol.tslocation = tslocation
end

function solution_new_original_retcode(
        sol::ODESolution{T, N}, original, retcode, resid) where {T, N}
    @reset sol.original = original
    @reset sol.retcode = retcode
    return @set sol.resid = resid
end

function solution_slice(sol::ODESolution{T, N}, I) where {T, N}
    @reset sol.u = sol.u[I]
    @reset sol.u_analytic = sol.u_analytic === nothing ? nothing : sol.u_analytic[I]
    @reset sol.t = sol.t[I]
    @reset sol.k = sol.dense ? sol.k[I] : sol.k
    return @set sol.alg = false
end

mask_discretes(::Nothing, _, _...) = nothing

function mask_discretes(
        discretes::ParameterTimeseriesCollection, new_t, ::Union{Int, CartesianIndex})
    masked_discretes = map(discretes) do disc
        i = searchsortedlast(disc.t, new_t)
        disc[i:i]
    end
    return ParameterTimeseriesCollection(masked_discretes, parameter_values(discretes))
end

function mask_discretes(discretes::ParameterTimeseriesCollection, new_t, ::AbstractRange)
    mint, maxt = extrema(new_t)
    masked_discretes = map(discretes) do disc
        mini = searchsortedfirst(disc.t, mint)
        maxi = searchsortedlast(disc.t, maxt)
        disc[mini:maxi]
    end
    return ParameterTimeseriesCollection(masked_discretes, parameter_values(discretes))
end

function mask_discretes(discretes::ParameterTimeseriesCollection, new_t, _)
    masked_discretes = map(discretes) do disc
        idxs = map(new_t) do t
            searchsortedlast(disc.t, t)
        end
        disc[idxs]
    end
    return ParameterTimeseriesCollection(masked_discretes, parameter_values(discretes))
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
    @reset sol.u = u
    @reset sol.t = t isa Vector ? t : collect(t)
    return @set sol.interp = interp
end

struct LazyInterpolationException <: Exception
    var::Symbol
end

function Base.showerror(io::IO, e::LazyInterpolationException)
    print(io, "The algorithm", e.var,
        " uses lazy interpolation, which is incompatible with `strip_solution`.")
end

function strip_solution(sol::ODESolution; strip_alg = false)
    if has_lazy_interpolation(sol.alg)
        throw(LazyInterpolationException(nameof(typeof(sol.alg))))
    end

    interp = strip_interpolation(sol.interp)

    @reset sol.interp = interp

    @reset sol.prob = (; p = nothing)

    if strip_alg
        @reset sol.alg = nothing
    end

    return sol
end
