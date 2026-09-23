@verbosity_specifier KeywordVerbosity begin
    toggles = (:undefined_keyword, :unhonored_keyword)
    presets = (
        None = (undefined_keyword = Silent, unhonored_keyword = Silent),
        Minimal = (undefined_keyword = WarnLevel, unhonored_keyword = WarnLevel),
        Standard = (undefined_keyword = WarnLevel, unhonored_keyword = WarnLevel),
        Detailed = (undefined_keyword = WarnLevel, unhonored_keyword = WarnLevel),
        All = (undefined_keyword = WarnLevel, unhonored_keyword = WarnLevel),
    )
    groups = ()
end

@doc """
    KeywordVerbosity(; undefined_keyword = WarnLevel, unhonored_keyword = WarnLevel)
    KeywordVerbosity(preset::SciMLLogging.AbstractVerbosityPreset)

SciMLLogging verbosity specifier for the problem-class keyword validation performed by
[`checkkwargs(prob, alg; kwargs...)`](@ref checkkwargs). It is passed to `solve`/`init`
as the `kwargs_verbosity` keyword:

```julia
using SciMLLogging: ErrorLevel
solve(prob, alg; maxtime = 10.0,
    kwargs_verbosity = SciMLBase.KeywordVerbosity(unhonored_keyword = ErrorLevel))
```

## Toggles

  - `undefined_keyword`: the user passed a keyword that has no meaning for the problem's
    class (for example `saveat` on a `NonlinearProblem`, or `reltol` on an
    `IntervalNonlinearProblem`). See [`keyword_class`](@ref).
  - `unhonored_keyword`: the user passed a keyword that is defined for the problem's
    class, but the chosen algorithm declares through [`can_honor`](@ref) that it cannot
    honor it.

Each toggle is a `SciMLLogging.MessageLevel`: `ErrorLevel` throws, `WarnLevel` and
`InfoLevel` log, `Silent` emits nothing. Both default to `WarnLevel`, as do the
`Minimal`, `Standard`, `Detailed` and `All` presets.

`KeywordVerbosity(SciMLLogging.None())` disables the class checks entirely, and takes
precedence over an explicit `kwargshandle`. This is the mechanism for solver-internal
`solve`/`init` calls that forward keywords from an enclosing solve (see
[`checkkwargs`](@ref)). A `Bool` `verbose` keyword never affects keyword validation.
""" KeywordVerbosity

const TOLERANCE_KEYWORDS = (
    :abstol, :reltol, :internalnorm, :xtol, :gtol, :constrtol, :compltol,
    :maxiters, :maxtime, :dtmin,
)

const TIMESERIES_KEYWORDS = (
    :dense, :saveat, :tstops, :d_discontinuities, :save_everystep, :save_on,
    :save_start, :save_end, :save_discretes, :initialize_save, :adaptive, :dt, :dtmax,
    :force_dtmin, :controller, :failfactor, :calck, :callback, :isoutofdomain,
    :unstable_check, :advance_to_tstop, :stop_at_next_tstop, :save_noise, :delta,
    :step_limiter, :stage_limiter, :timeseries_errors, :dense_errors,
    :weak_timeseries_errors, :weak_dense_errors,
)

const _NONTIMESERIES_UNDEFINED = TIMESERIES_KEYWORDS
# Optimization solvers take `callback` as a per-iteration callback.
const _OPTIMIZATION_UNDEFINED = filter(!=(:callback), TIMESERIES_KEYWORDS)

const KEYWORD_CLASSES = (
    ODE = (
        defined = (:abstol, :reltol, :internalnorm, :maxiters, :maxtime, :dtmin),
        undefined = (),
    ),
    DAE = (
        defined = (:abstol, :reltol, :internalnorm, :constrtol, :maxiters, :maxtime, :dtmin),
        undefined = (),
    ),
    BVP = (
        defined = (:abstol, :reltol, :internalnorm, :constrtol, :maxiters, :maxtime),
        undefined = (),
    ),
    SteadyState = (
        defined = (
            :abstol, :reltol, :internalnorm, :xtol, :gtol, :maxiters, :maxtime, :dtmin,
        ),
        undefined = (),
    ),
    Nonlinear = (
        defined = (:abstol, :reltol, :internalnorm, :xtol, :gtol, :maxiters, :maxtime),
        undefined = _NONTIMESERIES_UNDEFINED,
    ),
    NLLS = (
        defined = (:abstol, :reltol, :internalnorm, :xtol, :gtol, :maxiters, :maxtime),
        undefined = _NONTIMESERIES_UNDEFINED,
    ),
    Interval = (
        defined = (:abstol, :internalnorm, :maxiters, :maxtime),
        undefined = _NONTIMESERIES_UNDEFINED,
    ),
    Optimization = (
        defined = (
            :abstol, :reltol, :internalnorm, :xtol, :gtol, :constrtol, :compltol,
            :maxiters, :maxtime,
        ),
        undefined = _OPTIMIZATION_UNDEFINED,
    ),
    Linear = (
        defined = (:abstol, :reltol, :internalnorm, :maxiters, :maxtime),
        undefined = _NONTIMESERIES_UNDEFINED,
    ),
    Integral = (
        defined = (:abstol, :reltol, :internalnorm, :maxiters, :maxtime),
        undefined = _NONTIMESERIES_UNDEFINED,
    ),
    Discrete = (
        defined = (:internalnorm, :maxiters, :maxtime),
        undefined = (),
    ),
)

"""
    keyword_class(prob::AbstractSciMLProblem)::Union{Symbol, Nothing}

Return the problem class whose keyword matrix governs `solve`/`init` keywords for
`prob`, or `nothing` if the problem type is not classified (then only the unrecognized-
keyword check applies).

The classes and the tolerance/budget keywords (`abstol`, `reltol`, `internalnorm`,
`xtol`, `gtol`, `constrtol`, `compltol`, `maxiters`, `maxtime`, `dtmin`) each defines:

| class           | problem types                                                         | defined                                                                                      |
|:--------------- |:--------------------------------------------------------------------- |:-------------------------------------------------------------------------------------------- |
| `:ODE`          | ODE, SDE, RODE, DDE, SDDE (identity mass matrix), `ImplicitDiscreteProblem` | `abstol, reltol, internalnorm, maxiters, maxtime, dtmin`                                     |
| `:DAE`          | `AbstractDAEProblem`; the above with a non-`UniformScaling` mass matrix | `abstol, reltol, internalnorm, constrtol, maxiters, maxtime, dtmin`                        |
| `:BVP`          | `AbstractBVProblem`                                                   | `abstol, reltol, internalnorm, constrtol, maxiters, maxtime`                                 |
| `:SteadyState`  | `SteadyStateProblem`                                                  | `abstol, reltol, internalnorm, xtol, gtol, maxiters, maxtime, dtmin`                         |
| `:Nonlinear`    | other `AbstractNonlinearProblem`s                                     | `abstol, reltol, internalnorm, xtol, gtol, maxiters, maxtime`                                |
| `:NLLS`         | `NonlinearLeastSquaresProblem`                                        | `abstol, reltol, internalnorm, xtol, gtol, maxiters, maxtime`                                |
| `:Interval`     | `AbstractIntervalNonlinearProblem`                                    | `abstol, internalnorm, maxiters, maxtime`                                                    |
| `:Optimization` | `AbstractOptimizationProblem`                                         | `abstol, reltol, internalnorm, xtol, gtol, constrtol, compltol, maxiters, maxtime`           |
| `:Linear`       | `AbstractLinearProblem`                                               | `abstol, reltol, internalnorm, maxiters, maxtime`                                            |
| `:Integral`     | `AbstractIntegralProblem`                                             | `abstol, reltol, internalnorm, maxiters, maxtime`                                            |
| `:Discrete`     | `DiscreteProblem`; `AbstractJumpProblem` wrapping one                 | `internalnorm, maxiters, maxtime`                                                            |

A tolerance/budget keyword not listed for a class is undefined for it. In addition, the
time-series keywords (`dt`, `dtmax`, `saveat`, `tstops`, `dense`, `callback`,
`adaptive`, `controller`, and the other saving/stepping controls) are undefined for the
`:Nonlinear`, `:NLLS`, `:Interval`, `:Linear` and `:Integral` classes, and all but
`callback` are undefined for `:Optimization`. An `AbstractJumpProblem` takes the class
of the problem it wraps.

Packages defining new problem types outside SciMLBase extend this function.
See the RFC <https://github.com/SciML/SciMLBase.jl/issues/1562> for the meaning of
each keyword in each class.
"""
keyword_class(::AbstractSciMLProblem) = nothing
keyword_class(prob::AbstractODEProblem) = _timeseries_class(prob.f)
keyword_class(prob::AbstractSDEProblem) = _timeseries_class(prob.f)
keyword_class(prob::AbstractRODEProblem) = _timeseries_class(prob.f)
keyword_class(prob::AbstractDDEProblem) = _timeseries_class(prob.f)
keyword_class(prob::AbstractSDDEProblem) = _timeseries_class(prob.f)
keyword_class(::AbstractDAEProblem) = :DAE
keyword_class(::AbstractDiscreteProblem) = :Discrete
keyword_class(::ImplicitDiscreteProblem) = :ODE
keyword_class(::AbstractDynamicOptProblem) = nothing
keyword_class(::AbstractAnalyticalProblem) = nothing
keyword_class(prob::AbstractJumpProblem) = keyword_class(prob.prob)
keyword_class(::AbstractBVProblem) = :BVP
keyword_class(::AbstractNonlinearProblem) = :Nonlinear
keyword_class(::SteadyStateProblem) = :SteadyState
keyword_class(::NonlinearLeastSquaresProblem) = :NLLS
keyword_class(::AbstractIntervalNonlinearProblem) = :Interval
keyword_class(::AbstractOptimizationProblem) = :Optimization
keyword_class(::AbstractLinearProblem) = :Linear
keyword_class(::AbstractIntegralProblem) = :Integral

function _timeseries_class(f)
    hasproperty(f, :mass_matrix) || return :ODE
    return f.mass_matrix isa UniformScaling ? :ODE : :DAE
end

function _is_undefined_for_class(class::Symbol, kw::Symbol)
    spec = KEYWORD_CLASSES[class]
    kw in spec.undefined && return true
    return kw in TOLERANCE_KEYWORDS && !(kw in spec.defined)
end
_is_undefined_for_class(::Nothing, ::Symbol) = false

"""
    keyword_status(prob::AbstractSciMLProblem, kw::Symbol)::Symbol

Return `:defined`, `:undefined` (a real keyword with no meaning for the class of
`prob`, see [`keyword_class`](@ref)), or `:unrecognized` (not a common `solve`/`init`
keyword at all).
"""
function keyword_status(prob::AbstractSciMLProblem, kw::Symbol)
    _is_undefined_for_class(keyword_class(prob), kw) && return :undefined
    return kw in allowedkeywords || kw in TOLERANCE_KEYWORDS ? :defined : :unrecognized
end

"""
    can_honor(alg, ::Val{kw})::Bool

Declare whether algorithm `alg` honors the `solve`/`init` keyword `kw`. The default is
`true`. An algorithm that accepts a keyword defined for its problem class but ignores
it must return `false`, for example

```julia
SciMLBase.can_honor(::MyFixedStepAlg, ::Val{:abstol}) = false
SciMLBase.can_honor(::MyFixedStepAlg, ::Val{:maxtime}) = false
```

so that [`checkkwargs(prob, alg; kwargs...)`](@ref checkkwargs) reports an
`unhonored_keyword` message when the user passes it. It is only consulted for the
tolerance/budget keywords defined for the problem's [`keyword_class`](@ref).
"""
can_honor(alg, ::Val) = true

# `KeywordArgError` is the enum type itself, and is what callers pass for "error".
function _keyword_level(h)
    h === KeywordArgError && return ErrorLevel
    h == KeywordArgWarn && return WarnLevel
    h == KeywordArgSilent && return Silent
    throw(ArgumentError("`kwargshandle` must be `KeywordArgError`, `KeywordArgWarn` or `KeywordArgSilent`, got $h."))
end

_keyword_verbosity(kwargshandle, v::KeywordVerbosity{false}) = v
function _keyword_verbosity(kwargshandle, v)
    if kwargshandle !== nothing
        level = _keyword_level(kwargshandle)
        return KeywordVerbosity(undefined_keyword = level, unhonored_keyword = level)
    end
    v === nothing && return KeywordVerbosity()
    v isa KeywordVerbosity && return v
    throw(ArgumentError("`kwargs_verbosity` must be a `SciMLBase.KeywordVerbosity` or a SciMLLogging preset, got $(typeof(v))."))
end
_keyword_verbosity(kwargshandle, v::AbstractVerbosityPreset) =
    _keyword_verbosity(kwargshandle, KeywordVerbosity(v))

_join_keywords(kws) = join(("`$k`" for k in kws), ", ")

"""
    checkkwargs(kwargshandle; kwargs...)
    checkkwargs(prob::AbstractSciMLProblem, alg; kwargshandle = nothing,
        kwargs_verbosity = nothing, kwargs...)

Validate the keyword arguments of a user-facing `solve`/`init` call.

The one-argument form checks only that every keyword is in the common keyword list,
handling unrecognized keywords according to `kwargshandle` (a [`KeywordArgError`](@ref)
value).

The problem/algorithm form additionally checks the keywords against the keyword matrix
of the problem's class ([`keyword_class`](@ref)), reporting through SciMLLogging with
a [`KeywordVerbosity`](@ref):

 1. Unrecognized keywords are handled exactly as by `checkkwargs(kwargshandle; ...)`,
    with `kwargshandle = KeywordArgError` when it is `nothing`. The tolerance keywords
    `xtol`, `gtol`, `constrtol` and `compltol` are recognized here.
 2. Keywords undefined for the problem's class are reported through the
    `undefined_keyword` toggle.
 3. Tolerance/budget keywords that are defined for the class, but for which
    `can_honor(alg, Val(kw))` is `false`, are reported through the `unhonored_keyword`
    toggle.

Keywords whose value is `nothing` count as not passed for steps 2 and 3.

Severity precedence: `kwargs_verbosity = KeywordVerbosity(None())` disables steps 2 and
3 regardless of anything else. Otherwise an explicit `kwargshandle` wins
(`KeywordArgError → ErrorLevel`, `KeywordArgWarn → WarnLevel`,
`KeywordArgSilent → Silent`, applied to both toggles), then `kwargs_verbosity`, then
`KeywordVerbosity()`. Pass `kwargshandle` only when the user supplied it; `nothing`
means "not supplied". A `Bool` `verbose` keyword is ignored.

## Solver-internal calls

Validation applies only to the keywords a user passed to a user-facing call. A solver
that calls `solve`/`init` internally and forwards keywords from its own caller (for
example tolerances for a DAE initialization solve, or a steady-state solve that runs an
ODE solve) must pass `kwargs_verbosity = KeywordVerbosity(None())` on that internal
call, after any splatted user keywords so that it takes effect:

```julia
solve(inner_prob, inner_alg; abstol, reltol, kwargs...,
    kwargs_verbosity = SciMLBase.KeywordVerbosity(SciMLLogging.None()))
```
"""
function checkkwargs(
        prob::AbstractSciMLProblem, alg; kwargshandle = nothing,
        kwargs_verbosity = nothing, kwargs...
    )
    checkkwargs(
        kwargshandle === nothing ? KeywordArgError : kwargshandle;
        Base.structdiff(values(kwargs), NamedTuple{TOLERANCE_KEYWORDS})...
    )
    verbosity = _keyword_verbosity(kwargshandle, kwargs_verbosity)
    verbosity isa KeywordVerbosity{false} && return nothing

    class = keyword_class(prob)
    passed = filter(k -> kwargs[k] !== nothing, keys(kwargs))
    undefined = filter(k -> _is_undefined_for_class(class, k), passed)
    if !isempty(undefined)
        @SciMLMessage(
            lazy"Keyword(s) $(_join_keywords(undefined)) are not defined for a $(nameof(typeof(prob))) (problem class `$class`) and have no effect.",
            verbosity, :undefined_keyword
        )
    end
    unhonored = filter(passed) do k
        k in TOLERANCE_KEYWORDS && !_is_undefined_for_class(class, k) &&
            !can_honor(alg, Val(k))
    end
    if !isempty(unhonored)
        @SciMLMessage(
            lazy"Algorithm $(nameof(typeof(alg))) cannot honor keyword(s) $(_join_keywords(unhonored)); they are ignored.",
            verbosity, :unhonored_keyword
        )
    end
    return nothing
end
