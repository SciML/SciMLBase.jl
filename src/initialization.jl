"""
    $(TYPEDEF)

A collection of all the data required for `OverrideInit`.
"""
struct OverrideInitData{IProb, UIProb, IProbMap, IProbPmap}
    """
    The `AbstractNonlinearProblem` to solve for initialization.
    """
    initializeprob::IProb
    """
    A function which takes `(initializeprob, value_provider)` and updates
    the parameters of the former with their values in the latter.
    If absent (`nothing`) this will not be called, and the parameters
    in `initializeprob` will be used without modification. `value_provider`
    refers to a value provider as defined by SymbolicIndexingInterface.jl.
    Usually this will refer to a problem or integrator.
    """
    update_initializeprob!::UIProb
    """
    A function which takes the solution of `initializeprob` and returns
    the state vector of the original problem.
    """
    initializeprobmap::IProbMap
    """
    A function which takes the solution of `initializeprob` and returns
    the parameter object of the original problem. If absent (`nothing`),
    this will not be called and the parameters of the problem being
    initialized will be returned as-is.
    """
    initializeprobpmap::IProbPmap

    function OverrideInitData(initprob::I, update_initprob!::J, initprobmap::K,
            initprobpmap::L) where {I, J, K, L}
        @assert initprob isa Union{NonlinearProblem, NonlinearLeastSquaresProblem}
        return new{I, J, K, L}(initprob, update_initprob!, initprobmap, initprobpmap)
    end
end

"""
    get_initial_values(prob, valp, f, alg, isinplace; kwargs...)

Return the initial `u0` and `p` for the given SciMLProblem and initialization algorithm,
and a boolean indicating whether the initialization process was successful. Keyword
arguments to this function are dependent on the initialization algorithm. `prob` is only
required for dispatching. `valp` refers the appropriate data structure from which the
current state and parameter values should be obtained. `valp` is a non-timeseries value
provider as defined by SymbolicIndexingInterface.jl. `f` is the SciMLFunction for the
problem. `alg` is the initialization algorithm to use. `isinplace` is either `Val{true}`
if `valp` and the SciMLFunction are inplace, and `Val{false}` otherwise.
"""
function get_initial_values end

struct CheckInitFailureError <: Exception
    normresid::Any
    abstol::Any
end

function Base.showerror(io::IO, e::CheckInitFailureError)
    print(io,
        "CheckInit specified but initialization not satisfied. normresid = $(e.normresid) > abstol = $(e.abstol)")
end

struct OverrideInitMissingAlgorithm <: Exception end

function Base.showerror(io::IO, e::OverrideInitMissingAlgorithm)
    print(io,
        "OverrideInit specified but no NonlinearSolve.jl algorithm provided. Provide an algorithm via the `nlsolve_alg` keyword argument to `get_initial_values`.")
end

"""
Utility function to evaluate the RHS of the ODE, using the integrator's `tmp_cache` if
it is in-place or simply calling the function if not.
"""
function _evaluate_f_ode(integrator, f, isinplace::Val{true}, args...)
    tmp = first(get_tmp_cache(integrator))
    f(tmp, args...)
    return tmp
end

function _evaluate_f_ode(integrator, f, isinplace::Val{false}, args...)
    return f(args...)
end

"""
    $(TYPEDSIGNATURES)

A utility function equivalent to `Base.vec` but also handles `Number` and
`AbstractSciMLScalarOperator`.
"""
_vec(v) = vec(v)
_vec(v::Number) = v
_vec(v::SciMLOperators.AbstractSciMLScalarOperator) = v
_vec(v::AbstractVector) = v

"""
    $(TYPEDSIGNATURES)

Check if the algebraic constraints are satisfied, and error if they aren't. Returns
the `u0` and `p` as-is, and is always successful if it returns. Valid only for
`ODEProblem` and `DAEProblem`. Requires a `DEIntegrator` as its second argument.
"""
function get_initial_values(
        prob::AbstractDEProblem, integrator::DEIntegrator, f, alg::CheckInit,
        isinplace::Union{Val{true}, Val{false}}; kwargs...)
    u0 = state_values(integrator)
    p = parameter_values(integrator)
    t = current_time(integrator)
    M = f.mass_matrix

    algebraic_vars = [all(iszero, x) for x in eachcol(M)]
    algebraic_eqs = [all(iszero, x) for x in eachrow(M)]
    (iszero(algebraic_vars) || iszero(algebraic_eqs)) && return u0, p, true
    update_coefficients!(M, u0, p, t)
    tmp = _evaluate_f_ode(integrator, f, isinplace, u0, p, t)
    tmp .= ArrayInterface.restructure(tmp, algebraic_eqs .* _vec(tmp))

    normresid = integrator.opts.internalnorm(tmp, t)
    if normresid > integrator.opts.abstol
        throw(CheckInitFailureError(normresid, integrator.opts.abstol))
    end
    return u0, p, true
end

"""
Utility function to evaluate the RHS of the DAE, using the integrator's `tmp_cache` if
it is in-place or simply calling the function if not.
"""
function _evaluate_f_dae(integrator, f, isinplace::Val{true}, args...)
    tmp = get_tmp_cache(integrator)[2]
    f(tmp, args...)
    return tmp
end

function _evaluate_f_dae(integrator, f, isinplace::Val{false}, args...)
    return f(args...)
end

function get_initial_values(
        prob::AbstractDAEProblem, integrator::DEIntegrator, f, alg::CheckInit,
        isinplace::Union{Val{true}, Val{false}}; kwargs...)
    u0 = state_values(integrator)
    p = parameter_values(integrator)
    t = current_time(integrator)

    resid = _evaluate_f_dae(integrator, f, isinplace, integrator.du, u0, p, t)
    normresid = integrator.opts.internalnorm(resid, t)
    if normresid > integrator.opts.abstol
        throw(CheckInitFailureError(normresid, integrator.opts.abstol))
    end
    return u0, p, true
end

"""
    $(TYPEDSIGNATURES)

Solve a `NonlinearProblem`/`NonlinearLeastSquaresProblem` to obtain the initial `u0` and
`p`. Requires that `f` have the field `initialization_data` which is an `OverrideInitData`.
If the field is absent or the value is `nothing`, return `u0` and `p` successfully as-is.
The NonlinearSolve.jl algorithm to use must be specified through the `nlsolve_alg` keyword
argument, failing which this function will throw an error. The success value returned
depends on the success of the nonlinear solve.
"""
function get_initial_values(prob, valp, f, alg::OverrideInit,
        iip::Union{Val{true}, Val{false}}; nlsolve_alg = nothing, kwargs...)
    u0 = state_values(valp)
    p = parameter_values(valp)

    if !has_initialization_data(f)
        return u0, p, true
    end

    initdata::OverrideInitData = f.initialization_data
    initprob = initdata.initializeprob

    nlsolve_alg = something(nlsolve_alg, alg.nlsolve, Some(nothing))
    if nlsolve_alg === nothing && state_values(initprob) !== nothing
        throw(OverrideInitMissingAlgorithm())
    end

    if initdata.update_initializeprob! !== nothing
        initdata.update_initializeprob!(initprob, valp)
    end

    nlsol = solve(initprob, nlsolve_alg; abstol = alg.abstol)

    u0 = initdata.initializeprobmap(nlsol)
    if initdata.initializeprobpmap !== nothing
        p = initdata.initializeprobpmap(nlsol)
    end

    return u0, p, SciMLBase.successful_retcode(nlsol)
end
