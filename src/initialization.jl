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
    A function which takes `value_provider` and the solution of `initializeprob` and returns
    the parameter object of the original problem. If absent (`nothing`),
    this will not be called and the parameters of the problem being
    initialized will be returned as-is.
    """
    initializeprobpmap::IProbPmap

    function OverrideInitData(initprob::I, update_initprob!::J, initprobmap::K,
            initprobpmap::L) where {I, J, K, L}
        @assert initprob isa
                Union{SCCNonlinearProblem, NonlinearProblem, NonlinearLeastSquaresProblem}
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
    isdae::Bool
end

function Base.showerror(io::IO, e::CheckInitFailureError)
    print(io,
        """
        DAE initialization failed: your u0 did not satisfy the initialization requirements, \
        normresid = $(e.normresid) > abstol = $(e.abstol).
        """)

    if e.isdae
        print(io,
            """
            If you wish for the system to automatically change the algebraic variables to \
            satisfy the algebraic constraints, please pass `initializealg = BrownBasicInit()` \
            to solve (this option will require `using OrdinaryDiffEqNonlinearSolve`). If you \
            wish to perform an initialization on the complete u0, please pass \
            `initializealg = ShampineCollocationInit()` to `solve`. Note that initialization \
            can be a very difficult process for DAEs and in many cases can be numerically \
            intractable without symbolic manipulation of the system. For an automated \
            system that will generate numerically stable initializations, see \
            ModelingToolkit.jl structural simplification for more details.
            """)
    end
end

struct OverrideInitMissingAlgorithm <: Exception end

function Base.showerror(io::IO, e::OverrideInitMissingAlgorithm)
    print(io,
        "OverrideInit specified but no NonlinearSolve.jl algorithm provided. Provide an algorithm via the `nlsolve_alg` keyword argument to `get_initial_values`.")
end

struct OverrideInitNoTolerance <: Exception
    tolerance::Symbol
end

function Base.showerror(io::IO, e::OverrideInitNoTolerance)
    print(io,
        "Tolerances were not provided to `OverrideInit`. `$(e.tolerance)` must be provided as a keyword argument to `get_initial_values` or as a keyword argument to the `OverrideInit` constructor.")
end

"""
Utility function to evaluate the RHS, using the integrator's `tmp_cache` if
it is in-place or simply calling the function if not.
"""
function _evaluate_f(integrator, f, isinplace::Val{true}, args...)
    tmp = first(get_tmp_cache(integrator))
    f(tmp, args...)
    return tmp
end

function _evaluate_f(integrator, f, isinplace::Val{false}, args...)
    return f(args...)
end

"""
Utility function to evaluate the RHS, adding extra arguments (such as history function for
DDEs) wherever necessary.
"""
function evaluate_f(integrator::DEIntegrator, prob, f, isinplace, u, p, t)
    return _evaluate_f(integrator, f, isinplace, u, p, t)
end

function evaluate_f(
        integrator::DEIntegrator, prob::AbstractDAEProblem, f, isinplace, u, p, t)
    return _evaluate_f(integrator, f, isinplace, integrator.du, u, p, t)
end

function evaluate_f(
        integrator::AbstractDDEIntegrator, prob::AbstractDDEProblem, f, isinplace, u, p, t)
    return _evaluate_f(integrator, f, isinplace, u, get_history_function(integrator), p, t)
end

function evaluate_f(integrator::AbstractSDDEIntegrator,
        prob::AbstractSDDEProblem, f, isinplace, u, p, t)
    return _evaluate_f(integrator, f, isinplace, u, get_history_function(integrator), p, t)
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
`AbstractDEProblem` and `AbstractDAEProblem`. Requires a `DEIntegrator` as its second argument.

Keyword arguments:
- `abstol`: The absolute value below which the norm of the residual of algebraic equations
  should lie. The norm function used is `integrator.opts.internalnorm` if present, and
  `LinearAlgebra.norm` if not.
"""
function get_initial_values(
        prob::AbstractDEProblem, integrator::DEIntegrator, f, alg::CheckInit,
        isinplace::Union{Val{true}, Val{false}}; abstol, kwargs...)
    u0 = state_values(integrator)
    p = parameter_values(integrator)
    t = current_time(integrator)
    M = f.mass_matrix

    M == I && return u0, p, true
    algebraic_vars = [all(iszero, x) for x in eachcol(M)]
    algebraic_eqs = [all(iszero, x) for x in eachrow(M)]
    (iszero(algebraic_vars) || iszero(algebraic_eqs)) && return u0, p, true
    update_coefficients!(M, u0, p, t)
    tmp = evaluate_f(integrator, prob, f, isinplace, u0, p, t)
    tmp .= ArrayInterface.restructure(tmp, algebraic_eqs .* _vec(tmp))

    normresid = isdefined(integrator.opts, :internalnorm) ?
                integrator.opts.internalnorm(tmp, t) : norm(tmp)
    if normresid > abstol
        throw(CheckInitFailureError(normresid, abstol, true))
    end
    return u0, p, true
end

function get_initial_values(
        prob::AbstractDAEProblem, integrator::DEIntegrator, f, alg::CheckInit,
        isinplace::Union{Val{true}, Val{false}}; abstol, kwargs...)
    u0 = state_values(integrator)
    p = parameter_values(integrator)
    t = current_time(integrator)

    resid = evaluate_f(integrator, prob, f, isinplace, u0, p, t)
    normresid = isdefined(integrator.opts, :internalnorm) ?
                integrator.opts.internalnorm(resid, t) : norm(resid)

    if normresid > abstol
        throw(CheckInitFailureError(normresid, abstol, false))
    end
    return u0, p, true
end

"""
    $(TYPEDSIGNATURES)

Solve a `NonlinearProblem`/`NonlinearLeastSquaresProblem` to obtain the initial `u0` and
`p`. Requires that `f` have the field `initialization_data` which is an `OverrideInitData`.
If the field is absent or the value is `nothing`, return `u0` and `p` successfully as-is.

The success value returned depends on the success of the nonlinear solve.

Keyword arguments:
- `nlsolve_alg`: The NonlinearSolve.jl algorithm to use. If not provided, this function will
  throw an error.
- `abstol`, `reltol`: The `abstol` (`reltol`) to use for the nonlinear solve. The value
  provided to the `OverrideInit` constructor takes priority over this keyword argument.
  If the former is `nothing`, this keyword argument will be used. If it is also not provided,
  an error will be thrown.

In case the initialization problem is trivial, `nlsolve_alg`, `abstol` and `reltol` are
not required.
"""
function get_initial_values(prob, valp, f, alg::OverrideInit,
        iip::Union{Val{true}, Val{false}}; nlsolve_alg = nothing, abstol = nothing, reltol = nothing, kwargs...)
    u0 = state_values(valp)
    p = parameter_values(valp)

    if !has_initialization_data(f)
        return u0, p, true
    end

    initdata::OverrideInitData = f.initialization_data
    initprob = initdata.initializeprob

    if initdata.update_initializeprob! !== nothing
        initdata.update_initializeprob!(initprob, valp)
    end

    if is_trivial_initialization(initdata)
        nlsol = initprob
        success = true
    else
        nlsolve_alg = something(nlsolve_alg, alg.nlsolve, Some(nothing))
        if nlsolve_alg === nothing && state_values(initprob) !== nothing
            throw(OverrideInitMissingAlgorithm())
        end
        if alg.abstol !== nothing
            _abstol = alg.abstol
        elseif abstol !== nothing
            _abstol = abstol
        else
            throw(OverrideInitNoTolerance(:abstol))
        end
        if alg.reltol !== nothing
            _reltol = alg.reltol
        elseif reltol !== nothing
            _reltol = reltol
        else
            throw(OverrideInitNoTolerance(:reltol))
        end
        nlsol = solve(initprob, nlsolve_alg; abstol = _abstol, reltol = _reltol)
        success = SciMLBase.successful_retcode(nlsol)
    end

    u0 = initdata.initializeprobmap(nlsol)
    if initdata.initializeprobpmap !== nothing
        p = initdata.initializeprobpmap(valp, nlsol)
    end

    return u0, p, success
end

is_trivial_initialization(::Nothing) = true

function is_trivial_initialization(initdata::OverrideInitData)
    !(initdata.initializeprob isa NonlinearLeastSquaresProblem) &&
        state_values(initdata.initializeprob) === nothing
end

function is_trivial_initialization(f::AbstractSciMLFunction)
    has_initialization_data(f) && is_trivial_initialization(f.initialization_data)
end

function is_trivial_initialization(prob::AbstractSciMLProblem)
    is_trivial_initialization(prob.f)
end
