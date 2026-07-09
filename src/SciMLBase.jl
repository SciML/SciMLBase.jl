module SciMLBase
if isdefined(Base, :Experimental) &&
        isdefined(Base.Experimental, Symbol("@max_methods"))
    @eval Base.Experimental.@max_methods 1
end
using ConstructionBase: ConstructionBase, getproperties, setproperties
using RecipesBase: RecipesBase, @recipe, @series
using RecursiveArrayTools: RecursiveArrayTools, AbstractDiffEqArray,
    AbstractVectorOfArray, ArrayPartition, DiffEqArray,
    VectorOfArray, recursive_mean, vecarr_to_vectors,
    vecvecapply
using SciMLStructures: SciMLStructures
using SymbolicIndexingInterface: SymbolicIndexingInterface, ArraySymbolic,
    ContinuousTimeseries, NotSymbolic,
    NotTimeseries, ParameterIndexingProxy,
    ParameterTimeseriesCollection,
    ParameterTimeseriesIndex, ProblemState,
    ScalarSymbolic, SymbolCache, Timeseries,
    all_variable_symbols, current_time,
    default_values, get_all_timeseries_indexes,
    get_history_function, getname, getp, getsym,
    getu, hasname, independent_variable_symbols,
    is_markovian, is_observed, is_parameter,
    is_parameter_timeseries, is_time_dependent,
    is_timeseries_parameter, is_variable,
    parameter_index, parameter_symbols,
    parameter_values, remake_buffer,
    set_parameter!, set_state!, setsym,
    state_values, symbolic_container,
    symbolic_evaluate, symbolic_type,
    timeseries_parameter_index, variable_index,
    variable_symbols,
    with_updated_parameter_timeseries_values
using DocStringExtensions: DocStringExtensions, FIELDS, SIGNATURES, TYPEDEF,
    TYPEDFIELDS, TYPEDSIGNATURES
using LinearAlgebra: LinearAlgebra, I, det, norm
using Statistics: Statistics, mean, median
using Distributed: Distributed, CachingPool, myid, nworkers, pmap, workers
using Markdown: Markdown, @doc_str
using Printf: Printf, @printf
import Preferences
using PreallocationTools: get_tmp, DiffCache, FixedSizeDiffCache

import Logging, ArrayInterface, Random
import IteratorInterfaceExtensions
import CommonSolve: solve, init, step!, solve!
import FunctionWrappersWrappers
import RuntimeGeneratedFunctions
import EnumX
import ADTypes: ADTypes, AbstractADType
import Accessors: @set, @reset, @delete, @insert
import StaticArraysCore: StaticArraysCore, SArray
import Adapt: adapt_structure, adapt

using Reexport: Reexport, @reexport
using SciMLOperators: SciMLOperators
using SciMLOperators:
    AbstractSciMLOperator,
    IdentityOperator, NullOperator,
    InvertibleOperator, AbstractSciMLScalarOperator

import SciMLOperators:
    update_coefficients, update_coefficients!,
    isconstant, iscached, islinear, issquare,
    has_adjoint, has_expmv, has_expmv!, has_exp,
    has_mul, has_mul!, has_ldiv, has_ldiv!

@reexport using SciMLOperators

using SciMLPublic: @public

using SciMLLogging: @SciMLMessage, verbosity_to_bool

"""
    __solve(prob, alg, args...; kwargs...)

The low-level entry point that `CommonSolve.solve` forwards to after performing the
common pre-solve handling (such as argument checking and high-level error messages).
Solver packages add methods to `SciMLBase.__solve` dispatched on their problem and
algorithm types; this is the documented extension hook for implementing a solver.
Defining `__solve` rather than `solve` directly allows `SciMLBase.solve` to keep a
common implementation across all solvers.

Implementations should accept the same positional and keyword arguments that the
public `solve` method for the package accepts after `prob` and `alg`. They should
return the package's solution object and are responsible for honoring common
SciML keywords that apply to the problem family. Solver packages should dispatch
on concrete problem and algorithm types that they own, or on documented abstract
interfaces when that is the intended solver extension point, to avoid method
ambiguity with other packages.

See also [`__init`](@ref), which implements the reusable cache/iterator path.
"""
function __solve end

"""
    __init(prob, alg, args...; kwargs...)

The low-level entry point that `CommonSolve.init` forwards to after performing the
common pre-init handling (such as argument checking and high-level error messages).
Solver packages add methods to `SciMLBase.__init` dispatched on their problem and
algorithm types, returning the iterator/integrator object used by `solve!`; this is
the documented extension hook for implementing a solver. Defining `__init` rather than
`init` directly allows `SciMLBase.init` to keep a common implementation across all
solvers.

Implementations should construct and return the mutable cache, iterator, or
integrator object associated with the solver. That object should support
`solve!` and, when the corresponding algorithm trait returns `true`, direct
stepping through `step!`. Solver packages should keep `__init` keyword handling
consistent with `__solve` so that `solve(prob, alg; kwargs...)` and
`solve!(init(prob, alg; kwargs...))` agree when both paths are supported.

See also [`__solve`](@ref), which implements the direct solve path.
"""
function __init end

# Local alias for the `Union{Function, Type}` callable bound (mirrors the
# unexported `Base.Callable`), so problem constructors can dispatch on it
# without a non-public qualified access to Base.
const Callable = Union{Function, Type}

"""
$(TYPEDEF)

Base abstract type for all SciML problem definitions. A concrete
`AbstractSciMLProblem` encodes the mathematical data that a solver consumes via
`solve`, `init`, or lower-level extension methods.

## Interface

Concrete problem types should document the mathematical problem they encode, the
accepted constructor forms, and the fields that solvers may use. Common fields
are:

  - `f`: the function, operator, or symbolic interface defining the equations.
  - `u0`: the initial state, initial guess, or state-like data when one exists.
  - `tspan`: the independent-variable interval for time-dependent problems.
  - `p`: parameters, defaulting to `NullParameters` when omitted.
  - `kwargs`: keyword arguments stored on the problem and forwarded to solves.
  - `problem_type`: optional internal metadata preserving how a problem was
    originally constructed when several constructors share one concrete type.

Subtypes that expose state and parameters through symbolic indexing should
implement or delegate the `SymbolicIndexingInterface` methods. By default,
`prob[sym]` indexes state variables, `prob.ps[sym]` indexes parameters, and
`current_time(prob)` is `prob.tspan[1]` for time-dependent problems. Parameter
indexing through `prob[sym]` is reserved for state variables and errors so that
parameter access remains explicit through `prob.ps`.

Problem types with an in-place/out-of-place function distinction should encode
that choice in their type parameters and support [`isinplace`](@ref). Problem
types that allow solver aliasing should define a matching
[`AbstractAliasSpecifier`](@ref) subtype and document which stored arrays may be
aliased by solvers.
"""
abstract type AbstractSciMLProblem end

# Problems
"""
$(TYPEDEF)

Base type for differential equation problems. Concrete subtypes encode equations
whose state is advanced, constrained, or sampled over an independent-variable
domain.

## Interface

Differential equation problems generally provide `f`, `u0`, `tspan`, `p`, and
`kwargs` fields, and follow the [`AbstractSciMLProblem`](@ref) symbolic indexing
rules. The problem's function is usually a subtype of
[`AbstractDiffEqFunction`](@ref), or is converted to one by the concrete
constructor, and its in-place choice is returned by [`isinplace`](@ref).

Subtypes should document the mathematical form of their equation, the expected
call signatures of their functions, and any additional data such as mass
matrices, delays, noise processes, jumps, callbacks, or boundary conditions.
"""
abstract type AbstractDEProblem <: AbstractSciMLProblem end

"""
$(TYPEDEF)

Marker supertype for auxiliary differential equation elements that are stored
inside higher-level problem definitions rather than solved directly.
"""
abstract type DEElement end

"""
$(TYPEDEF)

Marker supertype for sensitivity-problem metadata associated with differential
equation definitions.
"""
abstract type DESensitivity end

"""
$(TYPEDEF)

Base interface for linear system problems.

Concrete subtypes define systems such as `A * u = b`, optionally with an initial
guess for iterative solvers. The `bType` parameter records the right-hand side
type, and `isinplace` records whether the linear operator may mutate supplied
storage.

## Interface

Linear problems should provide `A`, `b`, `u0`, `p`, and `kwargs` fields. Matrix
and operator-backed definitions should use public matrix/operator interfaces
such as `AbstractMatrix` or `AbstractSciMLOperator`. Symbolic linear problems can
provide a symbolic container through an `f` or `symbolic_interface` field so that
state, parameter, and observed-value indexing delegates through
`SymbolicIndexingInterface`.
"""
abstract type AbstractLinearProblem{bType, isinplace} <: AbstractSciMLProblem end

"""
$(TYPEDEF)

Base interface for eigenvalue problems.

Concrete subtypes define standard eigenvalue problems `A * v = lambda * v` or
generalized problems `A * v = lambda * B * v`.

## Interface

Eigenvalue problems should provide the operator `A`, an optional generalized
operator `B`, parameter storage `p`, optional initial guess `u0`, solver
selection metadata such as `num_eigenpairs`, `eigentarget`, and `shift`, plus
stored solver `kwargs`. Extra keyword arguments are forwarded to solvers.
"""
abstract type AbstractEigenvalueProblem <: AbstractSciMLProblem end

"""
$(TYPEDEF)

Base interface for integral and quadrature problems.

The `isinplace` parameter records whether the integrand writes into supplied
storage. Concrete subtypes should document the integration domain, measure, batch
or sampled semantics, and whether the integrand has the call signature
`f(x, p)` or an in-place equivalent.

Integral problems commonly provide `f`, a domain or bounds field, `p`, and
stored solver `kwargs`, and use `NullParameters` when parameters are omitted.
"""
abstract type AbstractIntegralProblem{isinplace} <: AbstractSciMLProblem end

"""
$(TYPEDEF)

Base interface for optimization problems.

The `isinplace` parameter records whether the objective or derivative callbacks
write into supplied storage. Concrete subtypes encode an objective function, an
initial optimizer, optional parameters, constraints, bounds, and solver keyword
arguments.

Optimization problems should provide symbolic indexing metadata through their
optimization function or cache when constructed from a symbolic system. Solvers
that support reusable caches should use [`AbstractOptimizationCache`](@ref).
"""
abstract type AbstractOptimizationProblem{isinplace} <: AbstractSciMLProblem end

"""
$(TYPEDEF)

Base interface for reusable optimization solver caches.

Concrete caches must at least hold the optimization function, typically
`f <: OptimizationFunction`, and parameter values `p`. Caches that support
reinitialization may additionally provide a `reinit_cache` with replacement `u0`
and `p` values; `reinit!` and symbolic parameter access delegate through those
fields when present.
"""
abstract type AbstractOptimizationCache end

"""
$(TYPEDEF)

Base interface for nonlinear solve problems `f(u, p) = 0`.

The `uType` parameter records the initial guess type, and `isinplace` records
whether the nonlinear function writes residuals into supplied storage.

## Interface

Concrete subtypes should provide `f`, `u0`, `p`, and `kwargs` fields, plus any
solver-relevant bounds or problem metadata. State symbolic indexing reads and
writes `u0`; parameter indexing is exposed through `prob.ps`. Constructors should
wrap bare callables in an [`AbstractNonlinearFunction`](@ref) subtype when
needed.
"""
abstract type AbstractNonlinearProblem{uType, isinplace} <: AbstractSciMLProblem end

"""
$(TYPEDEF)

Base interface for interval nonlinear problems. These problems search for a
zero of `f(t, p)` over an interval `tspan` rather than for a root near an
initial state `u0`.

Concrete subtypes should provide `f`, `tspan`, `p`, and `kwargs` fields. The
`isinplace` parameter records whether the function writes its value into supplied
storage, and the `uType` parameter is available for array-valued interval
residuals.
"""
abstract type AbstractIntervalNonlinearProblem{uType, isinplace} <:
AbstractNonlinearProblem{
    uType,
    isinplace,
} end
"""
$(TYPEDEF)

Base for types which define steady-state problems, i.e. finding the `u` for which
`du/dt = f(u, p, t) = 0`. This is a type alias for [`AbstractNonlinearProblem`](@ref),
since a steady state is the solution of the nonlinear system defined by the right-hand
side of the differential equation.
"""
const AbstractSteadyStateProblem{
    uType, isinplace,
} = AbstractNonlinearProblem{
    uType,
    isinplace,
}

"""
$(TYPEDEF)

Base interface for problems that directly solve or sample an
[`AbstractNoiseProcess`](@ref). Concrete noise problems should provide a `noise`
field, a `tspan`, and solver keyword arguments. Their in-place behavior delegates
to the stored noise process through [`isinplace`](@ref).
"""
abstract type AbstractNoiseProblem <: AbstractDEProblem end

"""
$(TYPEDEF)

Base interface for ordinary differential equation problems.

Concrete ODE problems encode equations of the form `du/dt = f(u, p, t)`, or a
mass-matrix variant represented by the function object. The `uType`, `tType`,
and `isinplace` parameters record the initial state, promoted time span, and
function mutation convention.

## Interface

ODE problems should provide `f`, `u0`, `tspan`, `p`, `kwargs`, and optionally
`problem_type`. The function should support either `f(u, p, t)` or
`f(du, u, p, t)` according to [`isinplace`](@ref). Stored keyword arguments such
as callbacks or tolerances are forwarded to solvers.
"""
abstract type AbstractODEProblem{uType, tType, isinplace} <: AbstractDEProblem end

"""
$(TYPEDEF)

Base interface for dynamical optimization problems represented through the ODE
problem hierarchy. Concrete subtypes follow the [`AbstractODEProblem`](@ref)
contract and add optimization-specific objective, control, or constraint metadata
in their concrete fields.
"""
abstract type AbstractDynamicOptProblem{uType, tType, isinplace} <:
AbstractODEProblem{uType, tType, isinplace} end

"""
$(TYPEDEF)

Base interface for discrete-time recurrence problems. Concrete subtypes follow
the [`AbstractODEProblem`](@ref) field conventions but interpret `tspan` as the
iteration or discrete independent-variable span and `f` as the state update map.
"""
abstract type AbstractDiscreteProblem{uType, tType, isinplace} <:
AbstractODEProblem{uType, tType, isinplace} end

"""
$(TYPEDEF)

Base interface for analytical problems. Analytical problems follow the
[`AbstractODEProblem`](@ref) contract while indicating that the solution is
defined directly from an analytical function rather than numerical time stepping.
"""
abstract type AbstractAnalyticalProblem{uType, tType, isinplace} <:
AbstractODEProblem{uType, tType, isinplace} end

"""
$(TYPEDEF)

Base interface for random ordinary differential equation problems.

RODE problems follow the differential equation problem conventions and add a
noise process that is available to the dynamics, typically through a function
signature involving `W(t)`. The `ND` parameter records the noise-rate prototype
or dimensionality metadata used to determine diagonal versus non-diagonal noise.
"""
abstract type AbstractRODEProblem{uType, tType, isinplace, ND} <: AbstractDEProblem end

"""
$(TYPEDEF)

Base interface for stochastic differential equation problems.

SDE problems provide a drift function, a noise function, an initial state,
parameters, a time span, and noise-process metadata. Concrete subtypes commonly
store the drift/noise pair as `f` and `g`, plus `noise`, `noise_rate_prototype`,
`seed`, and `kwargs`.

The `ND` parameter records the noise-rate prototype. When it is `Nothing`, the
problem is treated as diagonal noise by [`is_diagonal_noise`](@ref); otherwise
the prototype describes the shape of the noise-rate output.
"""
abstract type AbstractSDEProblem{uType, tType, isinplace, ND} <:
AbstractRODEProblem{uType, tType, isinplace, ND} end

"""
$(TYPEDEF)

Base interface for differential-algebraic equation problems.

Concrete DAE problems encode residual equations involving both `u` and `du`,
with initial guesses for each. The `uType`, `duType`, `tType`, and `isinplace`
parameters record the state, derivative state, time span, and residual mutation
convention.

DAE problem subtypes should document their residual signature, usually
`f(resid, du, u, p, t)` for in-place functions or `f(du, u, p, t)` for
out-of-place functions, and any initialization data used to make initial
conditions consistent.
"""
abstract type AbstractDAEProblem{uType, duType, tType, isinplace} <: AbstractDEProblem end

"""
$(TYPEDEF)

Base interface for delay differential equation problems.

Concrete DDE problems follow the differential equation problem conventions and
add history data plus lag metadata. The `lType` parameter records the lag
specification type, and the function should document how it queries the history
function, commonly through `h(p, t)` or a solver-provided history interface.
"""
abstract type AbstractDDEProblem{uType, tType, lType, isinplace} <: AbstractDEProblem end

"""
$(TYPEDEF)

Base interface for DDE problems whose delays are constant over the solve.
Concrete subtypes follow the [`AbstractDDEProblem`](@ref) contract and should
provide the constant lag collection used by discontinuity handling and method
selection.
"""
abstract type AbstractConstantLagDDEProblem{uType, tType, lType, isinplace} <:
AbstractDDEProblem{uType, tType, lType, isinplace} end

"""
$(TYPEDEF)

Base interface for second-order ODE problems. These problems are represented in
the ODE hierarchy for solver interoperability, but their constructors preserve
second-order structure through concrete fields or `problem_type` metadata.
"""
abstract type AbstractSecondOrderODEProblem{uType, tType, isinplace} <:
AbstractODEProblem{uType, tType, isinplace} end

"""
$(TYPEDEF)

Base interface for boundary value problems.

Concrete BVP problems follow the ODE problem conventions and add boundary
condition functions and boundary data. The `nlls` parameter records whether the
boundary conditions are interpreted in a nonlinear least-squares form.

Subtypes should document their differential equation function, boundary condition
signature, initial mesh or guess, parameter storage, and how boundary residuals
are laid out.
"""
abstract type AbstractBVProblem{uType, tType, isinplace, nlls} <:
AbstractODEProblem{uType, tType, isinplace} end

"""
$(TYPEDEF)

Base interface for jump process problems.

Jump problems wrap an inner SciML problem and a jump collection. Symbolic
indexing, parameter access, state access, and current time delegate to the inner
problem. Concrete subtypes should provide the wrapped problem, jump definitions,
aggregation metadata, RNG/seed handling, and solver keyword arguments.
"""
abstract type AbstractJumpProblem{P, J} <: AbstractDEProblem end

"""
$(TYPEDEF)

Base interface for stochastic delay differential equation problems.

SDDE problems combine the SDE and DDE contracts: they provide drift and noise
functions, history data, lag metadata, noise-process metadata, parameters, and a
time span. The `ND` parameter records noise-rate prototype metadata and the
`lType` parameter records delay metadata.
"""
abstract type AbstractSDDEProblem{uType, tType, lType, isinplace, ND} <: AbstractDEProblem end

"""
$(TYPEDEF)

Base interface for SDDE problems whose delays are constant over the solve.
Concrete subtypes follow the [`AbstractSDDEProblem`](@ref) contract and should
provide the constant lag collection used by discontinuity handling and method
selection.
"""
abstract type AbstractConstantLagSDDEProblem{uType, tType, lType, isinplace, ND} <:
AbstractSDDEProblem{uType, tType, lType, isinplace, ND} end

"""
$(TYPEDEF)

Base interface for partial differential equation problems.

SciMLBase only defines the high-level PDE problem interface. Concrete PDE
problem types or discretization packages should document the symbolic or
discrete PDE representation, independent/dependent variables, domains,
parameters, boundary/initial conditions, and the discretization metadata needed
to transform the PDE into solver-ready SciML problems.
"""
abstract type AbstractPDEProblem <: AbstractDEProblem end

# Algorithms
"""
$(TYPEDEF)

Base interface for solver algorithm objects. A concrete `AbstractSciMLAlgorithm`
selects the numerical method used by `solve`, `init`, or lower-level extension
methods for an `AbstractSciMLProblem`.

## Interface

Concrete algorithms should be lightweight configuration objects. Solver-specific
options belong in the algorithm constructor, while options shared by a whole
problem family stay as `solve` or `init` keyword arguments. Solver packages
normally implement dispatches such as:

```julia
CommonSolve.solve(prob::AbstractSciMLProblem, alg::AbstractSciMLAlgorithm;
    kwargs...)
```

Algorithms should implement the relevant trait methods in `alg_traits.jl` when
their behavior differs from the default, such as adaptivity, supported number
types, automatic-differentiation behavior, solver order, stochastic integral
interpretation, or support for the caching and stepping interfaces. `remake` can
be used internally by solver packages to replace algorithm components such as
automatic-differentiation chunk sizes, but it is not part of the public user API
for choosing methods.
"""
abstract type AbstractSciMLAlgorithm end

"""
$(TYPEDEF)

Base interface for differential equation solver algorithms. Concrete subtypes
dispatch on differential equation problem types and should document the equation
families, state types, callbacks, events, interpolation, and initialization
features they support.

Differential equation algorithms participate in common traits such as
`isadaptive`, `isdiscrete`, `allowscomplex`, `alg_order`,
`isautodifferentiable`, `forwarddiffs_model`, and
`allows_late_binding_tstops`. The default trait values are conservative, so
solver packages should override them for concrete algorithms where appropriate.
"""
abstract type AbstractDEAlgorithm <: AbstractSciMLAlgorithm end

"""
$(TYPEDEF)

Base interface for linear solve algorithms. Concrete subtypes select methods for
`AbstractLinearProblem` instances, including direct factorizations, iterative
methods, preconditioner choices, and matrix-free operator handling.

Algorithm-specific choices such as factorization strategy, Krylov options, or
preconditioner configuration should be constructor fields on the concrete
algorithm. Shared solve controls remain keyword arguments to the solve call.
"""
abstract type AbstractLinearAlgorithm <: AbstractSciMLAlgorithm end

"""
$(TYPEDEF)

Base interface for nonlinear solve algorithms. Concrete subtypes select methods
for `AbstractNonlinearProblem` instances and should document their support for
in-place residuals, Jacobian information, bounds, line searches, trust regions,
termination controls, and reusable caches.
"""
abstract type AbstractNonlinearAlgorithm <: AbstractSciMLAlgorithm end

"""
$(TYPEDEF)

Base interface for interval nonlinear solve algorithms. Concrete subtypes solve
`AbstractIntervalNonlinearProblem` instances by searching for zeros over a
provided interval, commonly using bracketing or interval-based methods.

Concrete algorithms should document whether they require a sign change, how they
handle array-valued residuals, and which termination tolerances or bracketing
assumptions they use.
"""
abstract type AbstractIntervalNonlinearAlgorithm <: AbstractSciMLAlgorithm end

"""
$(TYPEDEF)

Base interface for integral and quadrature algorithms. Concrete subtypes solve
`AbstractIntegralProblem` instances and should document their domain support,
adaptive or fixed-sample behavior, batching semantics, random number usage, and
whether in-place integrands are supported.
"""
abstract type AbstractIntegralAlgorithm <: AbstractSciMLAlgorithm end

"""
$(TYPEDEF)

Base interface for optimization algorithms. Concrete subtypes solve
`AbstractOptimizationProblem` instances and should document their support for
bounds, nonlinear constraints, callbacks, gradients, Hessians, constraint
Jacobians, and reusable caches.

Optimization solver packages should override the optimization capability traits
in `alg_traits.jl`, such as `allowsbounds`, `requiresgradient`, `allowsfg`,
`allowsfgh`, `allowsconstraints`, and `allowscallback`, when the defaults do not
describe a concrete algorithm.
"""
abstract type AbstractOptimizationAlgorithm <: AbstractDEAlgorithm end

"""
$(TYPEDEF)

Base interface for steady-state algorithms. Concrete subtypes solve
`AbstractSteadyStateProblem` instances, usually by reusing ODE time-stepping,
nonlinear solve, or specialized fixed-point machinery to find `du/dt = 0`.

Concrete algorithms should document whether they expect an ODE-style inner
algorithm, a nonlinear solver, or a direct steady-state method, and how common
termination tolerances are interpreted.
"""
abstract type AbstractSteadyStateAlgorithm <: AbstractDEAlgorithm end

"""
$(TYPEDEF)

Base interface for boundary value problem algorithms. Concrete subtypes solve
`AbstractBVProblem` instances and should document their collocation, shooting,
mesh-adaptation, nonlinear-solver, and boundary-residual layout assumptions.
"""
abstract type AbstractBVPAlgorithm <: AbstractDEAlgorithm end

"""
$(TYPEDEF)

Base interface for ordinary differential equation algorithms. Concrete subtypes
solve `AbstractODEProblem` instances and should document their order, adaptivity,
stiffness assumptions, dense-output support, callback/event support, and
compatibility with mass matrices or split problem formulations.
"""
abstract type AbstractODEAlgorithm <: AbstractDEAlgorithm end

"""
$(TYPEDEF)

Base interface for algorithms that preserve second-order ODE structure. Concrete
subtypes should document whether they operate on a native second-order
formulation or on a first-order transformed problem, and which callback,
interpolation, and mass-matrix features remain available.
"""
abstract type AbstractSecondOrderODEAlgorithm <: AbstractDEAlgorithm end

"""
$(TYPEDEF)

Base interface for random ordinary differential equation algorithms. Concrete
subtypes solve `AbstractRODEProblem` instances and should document the noise
process assumptions, interpolation behavior, and how random forcing is sampled
or queried during a step.
"""
abstract type AbstractRODEAlgorithm <: AbstractDEAlgorithm end

"""
$(TYPEDEF)

Base interface for stochastic differential equation algorithms. Concrete
subtypes solve `AbstractSDEProblem` instances and should document their
stochastic integral interpretation, supported noise structures, adaptive behavior
and any restrictions such as additive-noise or Wiener-only assumptions.

SDE algorithms should override `alg_interpretation`, and may need to override
`allows_non_wiener_noise` or `requires_additive_noise`.
"""
abstract type AbstractSDEAlgorithm <: AbstractDEAlgorithm end

"""
$(TYPEDEF)

Base interface for differential-algebraic equation algorithms. Concrete subtypes
solve `AbstractDAEProblem` instances and should document their DAE index
assumptions, mass-matrix or residual form, consistent-initial-condition handling,
and supported `DAEInitializationAlgorithm` choices.
"""
abstract type AbstractDAEAlgorithm <: AbstractDEAlgorithm end

"""
$(TYPEDEF)

Base interface for delay differential equation algorithms. Concrete subtypes
solve `AbstractDDEProblem` instances and should document their lag support,
history interpolation, discontinuity handling, and callback/event behavior.
"""
abstract type AbstractDDEAlgorithm <: AbstractDEAlgorithm end

"""
$(TYPEDEF)

Base interface for stochastic delay differential equation algorithms. Concrete
subtypes combine the delay and stochastic algorithm contracts, including support
for lag metadata, history interpolation, noise structures, stochastic integral
interpretation, and discontinuity handling.
"""
abstract type AbstractSDDEAlgorithm <: AbstractDEAlgorithm end

"""
$(TYPEDEF)

Base interface for ensemble execution algorithms. These algorithms choose how
many related problem solves are scheduled for an `AbstractEnsembleProblem`; they
do not choose the numerical method for an individual trajectory.

Concrete subtypes should document the execution backend, serialization
requirements, random number behavior, task or process scheduling, and any
limitations on callbacks, reductions, or output functions.
"""
abstract type EnsembleAlgorithm <: AbstractSciMLAlgorithm end

"""
$(TYPEDEF)

Base interface for sensitivity algorithms passed through the `sensealg` keyword
argument to `solve`. A `sensealg` value chooses the method used to differentiate
the solver call, such as forward sensitivity equations, adjoint sensitivity
equations, direct AD through solver operations, second-order methods, or
shadowing methods for long-time averages.

`SciMLBase` owns the lightweight dispatch interface and fallback errors, while
sensitivity packages provide the concrete algorithms and ChainRules definitions
that implement derivatives of `solve`. Concrete sensitivity algorithms should be
small configuration objects: they should describe how derivatives are computed,
not store the problem being differentiated.

The type parameters are part of the dispatch key for downstream sensitivity
packages. Concrete algorithms should document the meaning of those parameters,
the supported problem families, the differentiable quantities (`u0`, `p`, save
values, observables, or problem-specific data), the automatic-differentiation
backends they use, and any restrictions on callbacks, events, mutation,
interpolation, or saved output.
"""
abstract type AbstractSensitivityAlgorithm{CS, AD, FDT} <: AbstractSciMLAlgorithm end

"""
$(TYPEDEF)

Base interface for sensitivity algorithms that differentiate through, or
otherwise overload, solver behavior. Subtypes participate in the `solve`
automatic-differentiation path by providing rules that replace the default
fallbacks for forward-mode and/or reverse-mode differentiation.

Concrete subtypes should document whether they differentiate the numerical solver
directly, solve auxiliary sensitivity equations, or combine solver rules with an
AD backend. They should also specify which solver features remain differentiable,
how saved values and interpolation are treated, and what happens for unsupported
problem or callback features.
"""
abstract type AbstractOverloadingSensitivityAlgorithm{CS, AD, FDT} <:
AbstractSensitivityAlgorithm{CS, AD, FDT} end

"""
$(TYPEDEF)

Base interface for forward sensitivity algorithms. Forward sensitivity methods
propagate tangent information alongside the primal solve and are typically used
when the derivative seed dimension is modest or when the caller needs full
solution sensitivities.

Concrete subtypes should document how tangent information is initialized,
propagated, and returned; which inputs are differentiated; how parameters and
initial conditions are seeded; and which problem, parameter, event, and callback
features are supported. If the method relies on an AD backend for Jacobian-vector
products or local derivative calculations, that backend and its limitations
should be documented by the concrete algorithm.
"""
abstract type AbstractForwardSensitivityAlgorithm{CS, AD, FDT} <:
AbstractOverloadingSensitivityAlgorithm{CS, AD, FDT} end

"""
$(TYPEDEF)

Base interface for adjoint sensitivity algorithms. Adjoint methods implement the
reverse-mode differentiation path for `solve`, propagating cotangents from saved
solution values or user objectives back to differentiable problem data.

Concrete subtypes should document their adjoint equation, what primal-solve data
must be retained or recomputed, checkpointing behavior, interpolation
requirements, vector-Jacobian product backend, and assumptions about callbacks,
events, discontinuities, noise, and mutation. They should also state which
solution outputs and problem fields may receive gradients.
"""
abstract type AbstractAdjointSensitivityAlgorithm{CS, AD, FDT} <:
AbstractOverloadingSensitivityAlgorithm{CS, AD, FDT} end

"""
$(TYPEDEF)

Base interface for second-order sensitivity algorithms. These algorithms compute
Hessian, Hessian-vector, or related second-derivative information for solver
outputs or objectives.

Concrete subtypes should document the first-order sensitivity method or AD
backend they build on, the supported second-derivative product or materialized
array form, how seeds and cotangents are represented, and any restrictions on
problem types, callbacks, saved output, or nested AD backends.
"""
abstract type AbstractSecondOrderSensitivityAlgorithm{CS, AD, FDT} <:
AbstractOverloadingSensitivityAlgorithm{CS, AD, FDT} end

"""
$(TYPEDEF)

Base interface for shadowing sensitivity algorithms. Shadowing methods estimate
derivatives of long-time statistics or trajectory-dependent quantities in
dynamical systems where direct trajectory sensitivities may be unsuitable.

Concrete subtypes should document the dynamical-systems assumptions required by
the method, how trajectories and transients are selected, what objective or
statistic is differentiated, how tangent or adjoint shadowing directions are
computed, and which solver features, callbacks, and parameterizations are
supported.
"""
abstract type AbstractShadowingSensitivityAlgorithm{CS, AD, FDT} <:
AbstractOverloadingSensitivityAlgorithm{CS, AD, FDT} end

"""
$(TYPEDEF)

Base interface for DAE initialization algorithms selected by the `initializealg`
keyword. Concrete subtypes control how solvers check, skip, or repair initial
conditions before integration starts.

DAE initialization algorithms should document which problem metadata they need,
whether they modify `u0` or `du0`, and which tolerances or inner nonlinear
solvers they use.
"""
abstract type DAEInitializationAlgorithm <: AbstractSciMLAlgorithm end

"""
    struct NoInit <: DAEInitializationAlgorithm

An initialization algorithm that completely skips the initialization phase. The solver
will use the provided initial conditions directly without any consistency checks or
modifications.

!!! warning
    Using `NoInit()` with inconsistent initial conditions will likely cause
    solver failures or incorrect results. Only use this when you are absolutely certain
    your initial conditions satisfy all DAE constraints.

This is useful when:
- You know your initial conditions are already perfectly consistent
- You want to avoid the computational cost of initialization
- You are debugging solver issues and want to isolate initialization from integration

## Example
```julia
prob = DAEProblem(f, du0_consistent, u0_consistent, tspan)
sol = solve(prob, IDA(), initializealg = NoInit())
```
"""
struct NoInit <: DAEInitializationAlgorithm end

"""
    struct CheckInit <: DAEInitializationAlgorithm

An initialization algorithm that only checks if the initial conditions are consistent
with the DAE constraints, without attempting to modify them. If the conditions are not
consistent within the solver's tolerance, an error will be thrown.

This is useful when:
- You have already computed consistent initial conditions
- You want to verify the consistency of your initial guess
- You want to ensure no automatic modifications are made to your initial conditions

## Example
```julia
prob = DAEProblem(f, du0, u0, tspan)
sol = solve(prob, IDA(), initializealg = CheckInit())
```
"""
struct CheckInit <: DAEInitializationAlgorithm end

"""
    struct OverrideInit <: DAEInitializationAlgorithm

An initialization algorithm that uses a separate initialization problem to find
consistent initial conditions. This is typically used with ModelingToolkit.jl
which can generate specialized initialization problems based on the model structure.

When using `OverrideInit`, the problem must have `initialization_data` that contains
an `initializeprob` field with the initialization problem to solve.

This algorithm is particularly useful for:
- High-index DAEs that have been index-reduced
- Systems with complex initialization requirements
- ModelingToolkit models with custom initialization equations

## Fields
- `abstol`: Absolute tolerance for the initialization solver
- `reltol`: Relative tolerance for the initialization solver
- `nlsolve`: Nonlinear solver to use for initialization

## Example
```julia
# Typically used automatically with ModelingToolkit
@named sys = ODESystem(eqs, t, vars, params)
sys = structural_simplify(sys)
prob = DAEProblem(sys, [], (0.0, 1.0), [])
# Will automatically use OverrideInit if initialization_data exists
sol = solve(prob, IDA())
```
"""
struct OverrideInit{T1, T2, F} <: DAEInitializationAlgorithm
    abstol::T1
    reltol::T2
    nlsolve::F
end

function OverrideInit(; abstol = nothing, reltol = nothing, nlsolve = nothing)
    return OverrideInit(abstol, reltol, nlsolve)
end
OverrideInit(abstol) = OverrideInit(; abstol = abstol, nlsolve = nothing)

# PDE Discretizations

"""
$(TYPEDEF)

Base interface for discretization algorithms. Concrete subtypes transform
symbolic or high-level PDE/problem descriptions into solver-ready SciML
problems, typically through `discretize` and optionally `symbolic_discretize`.

Concrete discretizations should document the input problem or system types they
accept, the numerical discretization they apply, the generated problem type, and
the metadata needed to map the numerical solution back to the original variables
and domains.
"""
abstract type AbstractDiscretization <: AbstractSciMLAlgorithm end

# Discretization metadata
"""
$(TYPEDEF)

Base interface for metadata produced by a discretization. The `hasTime` type
parameter records whether the wrapped solution has an independent time axis.

Concrete metadata types should store enough information for PDE solution wrappers
to recover the original variables, domains, dependent-variable layout, and the
solver-ready problem or solution generated by the discretizer.
"""
abstract type AbstractDiscretizationMetadata{hasTime} end

# Monte Carlo Simulations
"""
$(TYPEDEF)

Base interface for ensemble problems.

An `AbstractEnsembleProblem` describes many related solves generated from a
template problem. Concrete subtypes should expose the template problem, a
trajectory-generation hook, an output hook, a batch reduction hook, and any
initial reduction state needed by ensemble solvers. The standard concrete
implementation is [`EnsembleProblem`](@ref).
"""
abstract type AbstractEnsembleProblem <: AbstractSciMLProblem end

"""
$(TYPEDEF)

Base interface for algorithms that estimate ensemble quantities during an
ensemble solve.

Concrete estimator algorithms can be used by ensemble workflows to decide when a
Monte Carlo estimate has converged, how many trajectories are required, or how
batch reductions should be interpreted. Subtypes should document the statistic
they estimate, the stopping criterion, and the reduction state they require.
"""
abstract type AbstractEnsembleEstimator <: AbstractSciMLAlgorithm end

export EnsembleProblem
export EnsembleSolution, EnsembleTestSolution, EnsembleSummary

"""
$(TYPEDEF)

Base interface for interpolation objects carried by SciML solutions. Concrete
interpolation types describe how saved values are reconstructed between stored
solution points.

Concrete subtypes should document the interpolation order, whether derivatives
are available, the cache data they need from the solution, and which
`interpolation` or `interpolation!` methods they implement. Interpolation
objects are usually solver-owned implementation details, but solution types use
this supertype to expose consistent stripping and summary behavior.
"""
abstract type AbstractDiffEqInterpolation end

"""
$(TYPEDEF)
"""
abstract type AbstractDEOptions end

"""
$(TYPEDEF)

Base interface for solver caches used by differential equation integrators.
Concrete caches hold reusable arrays, factorizations, random increments,
temporary workspaces, or other mutable state needed across steps.

Solver packages should keep cache fields internal to the concrete integrator and
expose only the public cache accessors that are safe for users or callbacks,
such as `get_tmp_cache`, `user_cache`, `full_cache`, and the state-specific cache
helpers.
"""
abstract type DECache end

"""
$(TYPEDEF)
"""
abstract type DECallback end

"""
$(TYPEDEF)
"""
abstract type AbstractContinuousCallback <: DECallback end

"""
$(TYPEDEF)
"""
abstract type AbstractDiscreteCallback <: DECallback end

# Integrators
"""
$(TYPEDEF)

Base interface for differential equation integrators returned by `init`.
Integrators are mutable iterator-like solver states that can be advanced by
`step!`, finished by `solve!`, inspected or modified by callbacks, and
reinitialized when the concrete solver supports it.

The type parameters record the algorithm type `Alg`, the in-place/out-of-place
function convention `IIP`, the state type `U`, and the independent-variable type
`T`. Concrete integrators commonly expose fields such as `u`, `t`, `p`, `f`,
`alg`, `opts`, and `sol`, and should implement the relevant methods from the
integrator interface: stepping, cache access, state/time mutation, saving,
symbolic indexing, error checking, and optional RNG/reinitialization support.
"""
abstract type DEIntegrator{Alg, IIP, U, T} end

"""
$(TYPEDEF)

Base interface for steady-state integrators. These are returned by steady-state
`init` methods when the solver supports an iterator or cache interface for
finding an equilibrium.

Concrete subtypes follow the `DEIntegrator` conventions, but their independent
variable type is `Nothing` because the solve target is a terminal steady state
rather than a time series.
"""
abstract type AbstractSteadyStateIntegrator{Alg, IIP, U} <:
DEIntegrator{Alg, IIP, U, Nothing} end

"""
$(TYPEDEF)

Base interface for ODE integrators. Concrete subtypes advance an
`AbstractODEProblem` with an `AbstractODEAlgorithm` and should implement the
standard differential equation integrator operations for stepping, interpolation,
callback handling, saving, and cache access.
"""
abstract type AbstractODEIntegrator{Alg, IIP, U, T} <: DEIntegrator{Alg, IIP, U, T} end

"""
$(TYPEDEF)

Base interface for second-order ODE integrators. Concrete subtypes preserve, or
wrap, second-order problem structure while following the standard `DEIntegrator`
stepping, callback, saving, and cache contracts.
"""
abstract type AbstractSecondOrderODEIntegrator{Alg, IIP, U, T} <:
DEIntegrator{Alg, IIP, U, T} end

"""
$(TYPEDEF)

Base interface for RODE integrators. Concrete subtypes advance random ordinary
differential equation problems and should document how they expose or update the
noise process during stepping and interpolation.
"""
abstract type AbstractRODEIntegrator{Alg, IIP, U, T} <: DEIntegrator{Alg, IIP, U, T} end

"""
$(TYPEDEF)

Base interface for SDE integrators. Concrete subtypes advance stochastic
differential equation problems and should implement the standard integrator
operations plus any RNG, noise-cache, stochastic-interpolation, and
noise-process access required by the solver.
"""
abstract type AbstractSDEIntegrator{Alg, IIP, U, T} <: DEIntegrator{Alg, IIP, U, T} end

"""
$(TYPEDEF)

Base interface for DDE integrators. Concrete subtypes advance delay differential
equation problems and should document their history interpolation, lag handling,
discontinuity tracking, callback behavior, and cache access.
"""
abstract type AbstractDDEIntegrator{Alg, IIP, U, T} <: DEIntegrator{Alg, IIP, U, T} end

"""
$(TYPEDEF)

Base interface for DAE integrators. Concrete subtypes advance differential-
algebraic equation problems and should document their residual form, derivative
state handling, initialization behavior, algebraic consistency checks, and
callback reinitialization support.
"""
abstract type AbstractDAEIntegrator{Alg, IIP, U, T} <: DEIntegrator{Alg, IIP, U, T} end

"""
$(TYPEDEF)

Base interface for SDDE integrators. Concrete subtypes combine the stochastic
and delay integrator contracts, including RNG/noise handling, history
interpolation, lag metadata, discontinuity tracking, and callback behavior.
"""
abstract type AbstractSDDEIntegrator{Alg, IIP, U, T} <: DEIntegrator{Alg, IIP, U, T} end

# Solutions
"""
$(TYPEDEF)

Abstract supertype for solutions whose result has no saved independent-variable
axis. Concrete subtypes represent terminal outputs such as linear solves,
nonlinear solves, integrals, optimization solves, and other problems where the
primary result is the object stored in `sol.u`.

## Interface

Concrete subtypes must provide a `u` field. The array interface forwards to this
field: `size(sol) == size(sol.u)`, `sol[i] == sol.u[i]`, multidimensional
integer indexing is forwarded to `sol.u`, `sol[:]` returns `sol.u[:]`, and
`A * sol` forwards to `A * sol.u` for matrix-like `A`.

Subtypes that support mutation through the solution object should make
`sol[i] = value` update `sol.u[i]`. Symbolic state indexing is available when the
associated problem or cache implements the `SymbolicIndexingInterface` metadata;
parameter indexing should use `sol.ps`.

Common optional fields are `retcode`, `prob`, `alg`, `resid`, `original`, and
`stats`. Solver-specific solution types should document which of these fields
they provide.
"""
abstract type AbstractNoTimeSolution{T, N} <: AbstractArray{T, N} end

"""
$(TYPEDEF)

Abstract supertype for solutions that save a time series or another
ordered independent-variable series. Concrete subtypes are array-like views of
the saved states and store the raw saved values in `sol.u` with matching
independent-variable values in `sol.t`.

## Interface

Concrete subtypes must provide `u` and `t` fields with matching saved-value
indices. `sol[j]` is the saved state at `sol.t[j]`. Component indexing places
state indices before the saved-value index, so `sol[i, j]` is the `i`th
component of `sol.u[j]`, `sol[i, :]` is the time series of the `i`th component,
and higher-dimensional states follow the same rule, e.g. `sol[i, k, j]`.

The array dimensions describe the component axes plus the saved-value axis. For
multi-component states this means `length(sol)` can differ from `length(sol.t)`;
use `length(sol.t)` or `eachindex(sol.t)` when iterating over saved times.

Time-series solutions should provide `prob`, `alg`, `interp`, `dense`,
`retcode`, and `stats` fields when those concepts apply. Dense or piecewise
interpolation is exposed through callable syntax such as `sol(t)` and
`sol(t; idxs = idxs)` when the stored interpolation supports it. Symbolic state,
observed-variable, and parameter access is delegated through the
`SymbolicIndexingInterface` metadata on the solution's problem, with
time-varying parameter support supplied through `sol.discretes` and
`sol.saved_subsystem` when present.
"""
abstract type AbstractTimeseriesSolution{T, N, A} <: AbstractDiffEqArray{T, N, A} end

"""
$(TYPEDEF)

Abstract supertype for solutions from ensemble solves. An ensemble solution
stores a collection of trajectory results or summary trajectories in `sol.u` and
uses the `RecursiveArrayTools.AbstractVectorOfArray` interface so trajectories
can be indexed and plotted as a single array-like object.

## Interface

Concrete subtypes must provide a `u` field containing the trajectory solutions or
trajectory-like arrays. Standard ensemble solutions also provide `elapsedTime`,
`converged`, and optionally `stats`. Calling an ensemble solution forwards the
call to each trajectory in `sol.u`, so `sol(args...; kwargs...)` returns the
collection `[trajectory(args...; kwargs...) for trajectory in sol.u]` when the
stored trajectories are callable.

Analysis and plotting utilities assume that each element of `sol.u` is either an
`AbstractSciMLSolution` or follows the corresponding array/callable solution
interface closely enough for the requested operation.
"""
abstract type AbstractEnsembleSolution{T, N, A} <: AbstractVectorOfArray{T, N, A} end

"""
$(TYPEDEF)

Abstract supertype for saved stochastic noise processes. Noise processes are
`AbstractDiffEqArray`s so that saved noise values can be indexed consistently
with differential equation solutions while also carrying enough state for
stochastic solvers to replay, interpolate, or extend the noise path.

## Interface

Concrete subtypes are expected to expose the saved noise values through the
`AbstractDiffEqArray` interface and to be callable at an independent-variable
value when the process supports interpolation. The `isinplace` parameter records
whether the process updates supplied storage in-place. Solver code may also rely
on noise-process fields such as the current time and current noise value, so
concrete noise process types should document their own state fields and
mutation/reset semantics.
"""
abstract type AbstractNoiseProcess{T, N, A, isinplace} <: AbstractDiffEqArray{T, N, A} end

"""
    AbstractSciMLSolution

Union of all base SciML solution interfaces:
[`AbstractTimeseriesSolution`](@ref), [`AbstractNoTimeSolution`](@ref),
[`AbstractEnsembleSolution`](@ref), and [`AbstractNoiseProcess`](@ref).

This is a union rather than an abstract supertype so each solution family can
subtype the appropriate array interface directly. Use it for dispatch that
accepts any SciML solution, and use the narrower abstract solution types when a
method requires a time-series, no-time, ensemble, or noise-process contract.
"""
const AbstractSciMLSolution = Union{
    AbstractTimeseriesSolution,
    AbstractNoTimeSolution,
    AbstractEnsembleSolution,
    AbstractNoiseProcess,
}

"""
$(TYPEDEF)

Abstract interface for no-time solutions of linear systems. Concrete subtypes
store the computed solution in `u` and should follow the
[`AbstractNoTimeSolution`](@ref) array-forwarding contract. Linear solve
solutions commonly include `resid`, `alg`, `retcode`, `iters`, `cache`, and
`stats` fields so callers can inspect convergence and reuse solver caches.
"""
abstract type AbstractLinearSolution{T, N} <: AbstractNoTimeSolution{T, N} end

"""
$(TYPEDEF)

Abstract interface for no-time eigenvalue problem solutions. Concrete subtypes
store the computed eigenvalues in `u`, follow the
[`AbstractNoTimeSolution`](@ref) contract, and should document where eigenvectors,
residuals, the original problem, algorithm, return code, and solver statistics
are stored.
"""
abstract type AbstractEigenvalueSolution{T, N} <: AbstractNoTimeSolution{T, N} end

"""
$(TYPEDEF)

Abstract interface for no-time nonlinear equation solutions. Concrete subtypes
store the root or fixed point in `u` and the residual in a solver-specific
`resid` field when available. These solutions follow the
[`AbstractNoTimeSolution`](@ref) contract and commonly provide `prob`, `alg`,
`retcode`, `original`, bracket endpoints for interval methods, and `stats`.
"""
abstract type AbstractNonlinearSolution{T, N} <: AbstractNoTimeSolution{T, N} end

"""
$(TYPEDEF)

Abstract interface for no-time integral or quadrature solutions. Concrete
subtypes store the estimated integral in `u`, follow the
[`AbstractNoTimeSolution`](@ref) contract, and commonly provide `resid`, `prob`,
`alg`, `retcode`, `chi`, and `stats` fields for solver diagnostics.
"""
abstract type AbstractIntegralSolution{T, N} <: AbstractNoTimeSolution{T, N} end

"""
$(TYPEDEF)

Abstract interface for no-time optimization solutions. Concrete subtypes store
the optimizer or minimizer in `u` and follow the
[`AbstractNoTimeSolution`](@ref) contract. Optimization solutions commonly expose
`alg`, `objective`, `retcode`, `original`, `stats`, and a cache that supplies the
problem function and parameters for symbolic indexing.
"""
abstract type AbstractOptimizationSolution{T, N} <: AbstractNoTimeSolution{T, N} end

"""
$(TYPEDEF)

Alias for nonlinear solutions used by steady-state solvers. A steady-state
solution represents a state `u` satisfying the nonlinear system induced by the
differential equation right-hand side, and therefore follows the
[`AbstractNonlinearSolution`](@ref) and [`AbstractNoTimeSolution`](@ref)
contracts.
"""
const AbstractSteadyStateSolution{T, N} = AbstractNonlinearSolution{T, N}

"""
$(TYPEDEF)

Abstract interface for analytical time-series solutions. These solutions follow
the [`AbstractTimeseriesSolution`](@ref) contract and are used when the solution
values are generated from or compared against an analytical representation of
the problem. Plotting and error-calculation code may use analytical solution
metadata stored on the concrete solution or its problem.
"""
abstract type AbstractAnalyticalSolution{T, N, S} <: AbstractTimeseriesSolution{T, N, S} end

"""
$(TYPEDEF)

Abstract interface for ordinary differential equation time-series solutions.
Concrete subtypes follow the [`AbstractTimeseriesSolution`](@ref) contract and
typically provide saved states `u`, saved times `t`, interpolation data `interp`,
the original problem `prob`, the algorithm `alg`, solver `stats`, a `retcode`,
and optional analytical values, dense-output data, discrete parameter
time-series, residuals, and wrapped-solver output.
"""
abstract type AbstractODESolution{T, N, S} <: AbstractTimeseriesSolution{T, N, S} end

# Needed for plot recipes
"""
$(TYPEDEF)

Abstract interface for delay differential equation time-series solutions. These
solutions follow the [`AbstractODESolution`](@ref) contract and add delay-system
semantics through their problem, history function, and interpolation data.
"""
abstract type AbstractDDESolution{T, N, S} <: AbstractODESolution{T, N, S} end

"""
$(TYPEDEF)

Abstract interface for random ordinary differential equation time-series
solutions. These solutions follow the [`AbstractODESolution`](@ref) contract and
also carry the saved or reconstructible noise path used by the RODE, commonly
through a field or callable object such as `sol.W`.
"""
abstract type AbstractRODESolution{T, N, S} <: AbstractODESolution{T, N, S} end

"""
$(TYPEDEF)

Abstract interface for differential-algebraic equation time-series solutions.
These solutions follow the [`AbstractODESolution`](@ref) contract while
representing states that satisfy both differential and algebraic residual
conditions. Concrete subtypes should document any stored residuals,
initialization data, or consistent-initial-condition metadata they expose.
"""
abstract type AbstractDAESolution{T, N, S} <: AbstractODESolution{T, N, S} end

"""
$(TYPEDEF)

Abstract interface for PDE solutions with a saved time axis. These solutions
follow the [`AbstractTimeseriesSolution`](@ref) contract and additionally carry
discretization metadata. Concrete subtypes should provide `disc_data`,
`original_sol`, `ivdomain`, `ivs`, and `dvs` fields so downstream discretizer
packages can recover the PDE variables, domains, and original discretized solve.
Callable interpolation is discretizer-specific and should be implemented by the
package that owns the metadata type.
"""
abstract type AbstractPDETimeSeriesSolution{T, N, S, D} <:
AbstractTimeseriesSolution{T, N, S} end

"""
$(TYPEDEF)

Abstract interface for PDE solutions without a saved time axis. These solutions
follow the [`AbstractNoTimeSolution`](@ref) contract and additionally carry
discretization metadata. Concrete subtypes should provide `disc_data`,
`original_sol`, `ivdomain`, `ivs`, and `dvs` fields so downstream discretizer
packages can recover the PDE variables, domains, and original discretized solve.
Callable evaluation is discretizer-specific and should be implemented by the
package that owns the metadata type.
"""
abstract type AbstractPDENoTimeSolution{T, N, S, D} <:
AbstractNoTimeSolution{T, N} end

"""
$(TYPEDEF)

Union of PDE solution interfaces, covering both
[`AbstractPDETimeSeriesSolution`](@ref) and [`AbstractPDENoTimeSolution`](@ref)
with matching element, dimension, saved-value, and discretization metadata
parameters. Dispatch on this union when code accepts either time-dependent or
time-independent PDE wrapper solutions.
"""
const AbstractPDESolution{
    T,
    N,
    S,
    D,
} = Union{
    AbstractPDETimeSeriesSolution{T, N, S, D},
    AbstractPDENoTimeSolution{T, N, S, D},
}

"""
$(TYPEDEF)

Abstract interface for time-series solutions that store sensitivity quantities.
Sensitivity solutions follow the [`AbstractTimeseriesSolution`](@ref) contract,
but their saved values represent derivatives or augmented sensitivity states
rather than only the primal state. Concrete subtypes should document which
sensitivity method produced the values, how the sensitivity axes are arranged in
`u`, and whether interpolation supports derivative queries.
"""
abstract type AbstractSensitivitySolution{T, N, S} <: AbstractTimeseriesSolution{T, N, S} end

# Misc

"""
$(TYPEDEF)

Base interface for callable model-function containers used by SciML problems.

The `iip` type parameter records the in-place convention for the primary model
function and is returned by [`isinplace`](@ref). Subtypes with `iip == true`
write their result into the first argument and usually return `nothing`; subtypes
with `iip == false` return the computed value. Concrete function wrappers should
be callable with the same signature as their stored model function, and should
store optional derivative, sparsity, symbolic, and initialization metadata in
fields that the public `has_*` traits can query.

Subtypes should document the primary call signature, any auxiliary callbacks such
as Jacobians or vector-Jacobian products, the meaning of each prototype field,
and which optional callbacks must follow the same in-place convention as the
primary function. When optional data is unavailable, constructors should either
omit the field for that subtype or store `nothing` so that the corresponding
trait returns `false`.
"""
abstract type AbstractSciMLFunction{iip} end

"""
$(TYPEDEF)

Base interface for function containers that define differential equation
dynamics.

Concrete subtypes describe right-hand sides, residuals, maps, boundary-condition
systems, stochastic drift/diffusion pairs, or related decompositions used by
[`AbstractDEProblem`](@ref) subtypes. They follow the [`AbstractSciMLFunction`](@ref)
in-place contract and additionally standardize optional differential-equation
metadata such as mass matrices, analytic solutions, Jacobians, time gradients,
Jacobian-vector products, vector-Jacobian products, W-factorization callbacks,
parameter Jacobians, sparsity/coloring prototypes, symbolic systems, and
initialization data. The exact primary call signature is determined by the
concrete subtype, for example ODE functions use `f(u, p, t)` or
`f(du, u, p, t)`, while DAE functions use residual signatures involving
`du`, `u`, `p`, and `t`.
"""
abstract type AbstractDiffEqFunction{iip} <:
AbstractSciMLFunction{iip} end

"""
$(TYPEDEF)

Base interface for integral and quadrature integrand containers.

Concrete subtypes wrap scalar, array-valued, or batched integrands. Out-of-place
integrands return the integrand value, while in-place integrands write into an
output container supplied by the algorithm. In-place integrands must carry an
`integrand_prototype` so solvers can allocate correctly typed temporary storage;
batched integrands reserve their last array dimension for the batch axis.
"""
abstract type AbstractIntegralFunction{iip} <:
AbstractSciMLFunction{iip} end

"""
$(TYPEDEF)

Base interface for optimization objective containers.

Concrete subtypes wrap scalar or multi-objective cost functions together with
optional derivatives, Hessian-vector products, constraint callbacks, sparsity
prototypes, coloring metadata, observed quantities, and symbolic systems. The
`iip` parameter records whether auxiliary derivative/constraint callbacks mutate
their first argument, while the objective itself follows the documented
`OptimizationFunction` call convention.
"""
abstract type AbstractOptimizationFunction{iip} <: AbstractSciMLFunction{iip} end

"""
$(TYPEDEF)

Base interface for objects that provide delay-equation history values.

History functions are queried for state values before the initial time and for
delayed arguments during DDE/SDDE solves. Implementations should support the
call signatures required by the corresponding problem type, commonly `h(p, t)`
or `h(out, p, t)`, optional derivative queries such as `h(p, t, Val{i})`, and
indexing keywords accepted by delay solvers. Returned values must match the
state shape expected by the problem's function.
"""
abstract type AbstractHistoryFunction end

"""
$(TYPEDEF)
"""
abstract type AbstractReactionNetwork end

"""
$(TYPEDEF)

Internal. Used for signifying which AD context a derivative calculation is in.
"""
abstract type ADOriginator end

"""
$(TYPEDEF)

Base interface for solver aliasing policies.

Alias specifiers are passed through the common `alias` keyword to tell a solver
whether it may keep references to problem inputs instead of copying them into
solver-owned storage. Concrete specifiers use `Union{Bool, Nothing}` fields:
`true` permits aliasing, `false` requests non-aliasing behavior, and `nothing`
delegates that decision to the solver's default. Constructors that accept
`alias = true` or `alias = false` apply that value to every aliasable field of
the concrete specifier.

Aliasing is a performance and ownership hint. A solver may still copy data when
the selected algorithm requires an internal layout or when preserving correctness
requires solver-owned storage.
"""
abstract type AbstractAliasSpecifier end

"""
$(TYPEDEF)

Internal. Used for signifying the AD context comes from a ChainRules.jl definition.
"""
struct ChainRulesOriginator <: ADOriginator end

"""
$(TYPEDEF)

Internal. Used for signifying the AD context comes from an Enzyme.jl definition.
"""
struct EnzymeOriginator <: ADOriginator end

"""
$(TYPEDEF)

Internal. Used for signifying the AD context comes from a ReverseDiff.jl context.
"""
struct ReverseDiffOriginator <: ADOriginator end

"""
$(TYPEDEF)

Internal. Used for signifying the AD context comes from a Tracker.jl context.
"""
struct TrackerOriginator <: ADOriginator end

"""
$(TYPEDEF)

Internal. Used for signifying the AD context comes from a Mooncake.jl context.
"""
struct MooncakeOriginator <: ADOriginator end

include("initialization.jl")
include("odenlstep.jl")
include("utils.jl")
include("function_wrappers.jl")
include("scimlfunctions.jl")
include("alg_traits.jl")
include("debug.jl")

"""
    unwrapped_f(f)

Return the underlying user function with any function-wrapper layers removed. When `f`
has been wrapped (e.g. by `FunctionWrapperSpecialize` specialization, which specializes
`f` to a fixed `(u, p, t)` signature via a `FunctionWrappersWrapper`), this recovers the
original unwrapped function so it can be called on other argument types. If `f` is not
wrapped, it is returned unchanged.
"""
unwrapped_f(f) = f
unwrapped_f(f::Void) = unwrapped_f(f.f)
function unwrapped_f(f::FunctionWrappersWrappers.FunctionWrappersWrapper)
    return unwrapped_f(f.fw[1].obj[])
end

function specialization(
        ::Union{
            ODEFunction{iip, specialize},
            SDEFunction{iip, specialize}, DDEFunction{iip, specialize},
            SDDEFunction{iip, specialize},
            DAEFunction{iip, specialize},
            DynamicalODEFunction{iip, specialize},
            SplitFunction{iip, specialize},
            DynamicalSDEFunction{iip, specialize},
            SplitSDEFunction{iip, specialize},
            DynamicalDDEFunction{iip, specialize},
            DiscreteFunction{iip, specialize},
            ImplicitDiscreteFunction{iip, specialize},
            RODEFunction{iip, specialize},
            NonlinearFunction{iip, specialize},
            OptimizationFunction{iip, specialize},
            BVPFunction{iip, specialize},
            DynamicalBVPFunction{iip, specialize},
            IntegralFunction{iip, specialize},
            BatchIntegralFunction{iip, specialize},
        }
    ) where {
        iip,
        specialize,
    }
    return specialize
end

specialization(f::AbstractSciMLFunction) = FullSpecialize

"""
$(TYPEDEF)

Compatibility supertype for ODE-like function containers that carry explicit
parameterization metadata. New implementations should generally subtype a more
specific [`AbstractODEFunction`](@ref) wrapper and expose symbolic or parameter
metadata through public fields and `SymbolicIndexingInterface` methods.
"""
abstract type AbstractParameterizedFunction{iip} <: AbstractODEFunction{iip} end

include("retcodes.jl")
include("errors.jl")
include("symbolic_utils.jl")
include("performance_warnings.jl")

include("problems/discrete_problems.jl")
include("problems/implicit_discrete_problems.jl")
include("problems/steady_state_problems.jl")
include("problems/analytical_problems.jl")
include("problems/linear_problems.jl")
include("problems/eigenvalue_problems.jl")
include("problems/nonlinear_problems.jl")
include("problems/integral_problems.jl")
include("problems/ode_problems.jl")
include("problems/rode_problems.jl")
include("problems/sde_problems.jl")
include("problems/noise_problems.jl")
include("problems/bvp_problems.jl")
include("problems/dae_problems.jl")
include("problems/dde_problems.jl")
include("problems/sdde_problems.jl")
include("problems/pde_problems.jl")
include("problems/problem_utils.jl")
include("problems/problem_traits.jl")
include("problems/problem_interface.jl")
include("problems/optimization_problems.jl")

include("clock.jl")
include("solutions/save_idxs.jl")
include("solutions/basic_solutions.jl")
include("solutions/nonlinear_solutions.jl")
include("solutions/ode_solutions.jl")
include("solutions/rode_solutions.jl")
include("solutions/optimization_solutions.jl")
include("solutions/dae_solutions.jl")
include("solutions/pde_solutions.jl")
include("solutions/solution_interface.jl")

include("ensemble/ensemble_solutions.jl")
include("ensemble/ensemble_problems.jl")
include("ensemble/basic_ensemble_solve.jl")
include("ensemble/ensemble_analysis.jl")

include("solve.jl")
include("interpolation.jl")
include("integrator_interface.jl")
include("remake.jl")
include("callbacks.jl")

include("adapt.jl")

import PrecompileTools

PrecompileTools.@compile_workload begin
    # ODE test functions
    function lorenz(du, u, p, t)
        du[1] = 10.0(u[2] - u[1])
        du[2] = u[1] * (28.0 - u[3]) - u[2]
        du[3] = u[1] * u[2] - (8 / 3) * u[3]
    end

    function lorenz_oop(u, p, t)
        [10.0(u[2] - u[1]), u[1] * (28.0 - u[3]) - u[2], u[1] * u[2] - (8 / 3) * u[3]]
    end

    u0 = [1.0, 0.0, 0.0]
    tspan = (0.0, 1.0)

    # ODEProblem (IIP and OOP)
    prob_ode_iip = ODEProblem(lorenz, u0, tspan)
    ODEProblem(lorenz, u0, tspan, Float64[])
    prob_ode_oop = ODEProblem(lorenz_oop, u0, tspan)
    ODEProblem(lorenz_oop, u0, tspan, Float64[])

    # NonlinearProblem (IIP and OOP)
    nl_iip(res, u, p) = (res[1] = u[1]^2 - 2.0; nothing)
    nl_oop(u, p) = [u[1]^2 - 2.0]
    prob_nl_iip = NonlinearProblem(nl_iip, [1.0], nothing)
    prob_nl_oop = NonlinearProblem(nl_oop, [1.0], nothing)
    NonlinearProblem(nl_iip, [1.0])
    NonlinearProblem(nl_oop, [1.0])

    # OptimizationProblem
    opt_f = OptimizationFunction((x, p) -> sum(abs2, x))
    prob_opt = OptimizationProblem(opt_f, [1.0, 2.0], nothing)
    OptimizationProblem((x, p) -> sum(abs2, x), [1.0, 2.0])

    # IntegralProblem
    int_f = (x, p) -> x^2
    prob_int = IntegralProblem(int_f, (0.0, 1.0), nothing)
    IntegralProblem(int_f, (0.0, 1.0))

    # LinearProblem
    A = [1.0 2.0; 3.0 4.0]
    b = [1.0, 2.0]
    prob_lin = LinearProblem(A, b)
    LinearProblem(A, b, nothing)

    # SDEProblem
    g_iip(du, u, p, t) = (du .= 0.1; nothing)
    g_oop(u, p, t) = fill(0.1, length(u))
    prob_sde = SDEProblem(lorenz, g_iip, u0, tspan)
    SDEProblem(lorenz_oop, g_oop, u0, tspan)

    # DAEProblem
    dae_f(res, du, u, p, t) = (res[1] = du[1] - u[1]; nothing)
    DAEProblem(dae_f, [0.0], [1.0], (0.0, 1.0))

    # DDEProblem
    dde_f(du, u, h, p, t) = (du[1] = h(p, t - 0.1)[1]; nothing)
    h_func(p, t) = [1.0]
    DDEProblem(dde_f, [1.0], h_func, (0.0, 1.0))

    # SteadyStateProblem
    SteadyStateProblem(prob_ode_iip)
    SteadyStateProblem(prob_ode_oop)

    # DiscreteProblem
    discrete_f(du, u, p, t) = (du[1] = u[1] * 2; nothing)
    DiscreteProblem(discrete_f, [1.0], (0, 10))

    # remake for common problem types
    remake(prob_ode_iip, u0 = [2.0, 0.0, 0.0])
    remake(prob_ode_oop, u0 = [2.0, 0.0, 0.0])
    remake(prob_nl_iip, u0 = [2.0])
    remake(prob_nl_oop, u0 = [2.0])
    remake(prob_opt, u0 = [2.0, 3.0])
    remake(prob_lin, b = [2.0, 3.0])
    remake(prob_sde, u0 = [2.0, 0.0, 0.0])

    # RODEProblem
    rode_f(du, u, p, t, W) = (du .= u .* W; nothing)
    RODEProblem(rode_f, u0, tspan)

    # BVProblem
    bvp_f(du, u, p, t) = (du[1] = u[2]; du[2] = -u[1]; nothing)
    bvp_bc(res, u, p, t) = (res[1] = u[1][1] - 1.0; res[2] = u[2][1]; nothing)
    BVProblem(bvp_f, bvp_bc, [1.0, 0.0], (0.0, 1.0))

    # SecondOrderODEProblem
    so_f(ddu, du, u, p, t) = (ddu .= -u; nothing)
    SecondOrderODEProblem(so_f, [0.0], [1.0], (0.0, 1.0))

    # SplitODEProblem
    split_f1(du, u, p, t) = (du .= u; nothing)
    split_f2(du, u, p, t) = (du .= -u; nothing)
    SplitODEProblem(SplitFunction(split_f1, split_f2), [1.0], (0.0, 1.0))

    # ImplicitDiscreteProblem
    impl_f(res, u_next, u, p, t) = (res .= u_next .- 2 .* u; nothing)
    ImplicitDiscreteProblem(impl_f, [1.0], (0, 10))

    # NonlinearLeastSquaresProblem
    nllsq_f(u, p) = [u[1]^2 - 1.0]
    NonlinearLeastSquaresProblem(nllsq_f, [1.5])

    # IntervalNonlinearProblem
    interval_f(u, p) = u^2 - 2
    IntervalNonlinearProblem(interval_f, (0.0, 2.0))

    # Callbacks
    cb_condition(u, t, integrator) = true
    cb_affect!(integrator) = nothing
    DiscreteCallback(cb_condition, cb_affect!)

    ccb_condition(u, t, integrator) = t - 0.5
    ccb_affect!(integrator) = nothing
    ContinuousCallback(ccb_condition, ccb_affect!)

    cb = DiscreteCallback(cb_condition, cb_affect!)
    ccb = ContinuousCallback(ccb_condition, ccb_affect!)
    CallbackSet(cb, ccb)
end

"""
    discretize(sys, discretizer, args...; kwargs...)

Transform a symbolic or high-level problem description into a solver-ready
representation.

Discretizer packages implement `SciMLBase.discretize` for the system/problem
types and discretization algorithms they own. A method may return an
[`AbstractSciMLProblem`](@ref), a lower-level symbolic system, or another
documented object that downstream solver packages can consume. Implementations
should document the accepted input system type, the meaning of `discretizer`, the
generated problem family, and any metadata needed to map numerical solutions back
to the original variables and domains.

Use [`symbolic_discretize`](@ref) when the caller needs a diagnostic or symbolic
view of the discretized system rather than the solver-ready problem.
"""
function discretize end

"""
    symbolic_discretize(sys, discretizer, args...; kwargs...)

Return a symbolic or diagnostic representation of a discretization.

Discretizer packages implement `SciMLBase.symbolic_discretize` when they can
expose lowered equations, operators, grids, boundary-condition handling, or other
intermediate artifacts without constructing only the final solver-ready problem.
The result should correspond to the same mathematical discretization used by
[`discretize`](@ref), but may preserve additional symbolic information useful for
inspection, debugging, code generation, or downstream transformations.
"""
function symbolic_discretize end

isfunctionwrapper(x) = false
function wrapfun_oop end
function wrapfun_iip end
function unwrap_fw end

export ReturnCode


# Exports
export AllObserved

export isinplace

export solve, solve!, init, discretize, symbolic_discretize

export LinearProblem, LinearSolution, IntervalNonlinearProblem,
    IntegralProblem, IntegralSolution, SampledIntegralProblem,
    OptimizationProblem, OptimizationSolution

export EigenvalueProblem, EigenvalueSolution, EigenvalueTarget

export NonlinearProblem, NonlinearSolution,
    SCCNonlinearProblem, NonlinearLeastSquaresProblem, HomotopyProblem

export DiscreteProblem, ImplicitDiscreteProblem
export SteadyStateProblem, SteadyStateSolution
export NoiseProblem
export ODEProblem, ODESolution
export DynamicalODEFunction,
    DynamicalODEProblem,
    SecondOrderODEProblem, SplitFunction, SplitODEProblem
export SplitSDEProblem
export DynamicalSDEFunction, DynamicalSDEProblem
export RODEProblem, RODESolution, SDEProblem
export DAEProblem, DAESolution
export DDEProblem
export DynamicalDDEFunction, DynamicalDDEProblem,
    SecondOrderDDEProblem
export SDDEProblem
export PDEProblem, PDETimeSeriesSolution, PDENoTimeSolution
export IncrementingODEProblem

export BVProblem, TwoPointBVProblem, SecondOrderBVProblem, TwoPointSecondOrderBVProblem

export remake

export ODEFunction, DiscreteFunction, ImplicitDiscreteFunction, SplitFunction, DAEFunction,
    DDEFunction, SDEFunction, SplitSDEFunction, RODEFunction, SDDEFunction,
    IncrementingODEFunction, NonlinearFunction, HomotopyNonlinearFunction,
    IntervalNonlinearFunction, BVPFunction,
    TwoPointBVPFunction, TwoPointDynamicalBVPFunction,
    DynamicalBVPFunction, IntegralFunction, BatchIntegralFunction, ODEInputFunction

export OptimizationFunction, MultiObjectiveOptimizationFunction

export CheckInit

export EnsembleThreads, EnsembleDistributed, EnsembleSplitThreads, EnsembleSerial,
    EnsembleContext

export EnsembleAnalysis, EnsembleSummary


export step!, deleteat!, addat!, get_tmp_cache,
    full_cache, user_cache, u_cache, du_cache,
    rand_cache, ratenoise_cache,
    resize_non_user_cache!, deleteat_non_user_cache!, addat_non_user_cache!,
    terminate!,
    add_tstop!, has_tstop, first_tstop, pop_tstop!,
    add_saveat!, set_abstol!,
    set_reltol!, get_du, get_du!, get_dt, get_proposed_dt, set_proposed_dt!,
    derivative_discontinuity!, u_modified!, savevalues!, reinit!, auto_dt_reset!, set_t!,
    set_u!, check_error, change_t_via_interpolation!, addsteps!,
    isdiscrete, reeval_internals_due_to_modification!,
    has_rng, get_rng, set_rng!, supports_solve_rng

export ContinuousCallback, DiscreteCallback, CallbackSet, VectorContinuousCallback

export Clocks, TimeDomain, is_discrete_time_domain, isclock, issolverstepclock, iscontinuous

export ODEAliasSpecifier, LinearAliasSpecifier

# Public traits

@public has_init, has_step, successful_retcode

# Abstract interface types
@public AbstractSciMLProblem, AbstractSciMLSolution, AbstractDEAlgorithm,
    AbstractODEProblem, AbstractODEAlgorithm, AbstractNonlinearProblem,
    AbstractDynamicalODEProblem, AbstractIntegralAlgorithm, AbstractAliasSpecifier

# Solution / problem support types
@public NLStats, NullParameters, AutoSpecialize

# Core functions
@public build_solution, numargs

# SciMLFunction derivative traits
@public has_jac, has_jvp, has_vjp, has_tgrad, has_analytic, has_reinit,
    has_initialization_data, has_stats

# Function-argument validation errors
@public FunctionArgumentsError, TooFewArgumentsError, TooManyArgumentsError

# Abstract problem types
@public AbstractDEProblem, AbstractDAEProblem, AbstractSDEProblem, AbstractDDEProblem,
    AbstractDiscreteProblem, AbstractNoiseProblem, AbstractJumpProblem,
    AbstractEnsembleProblem

# Abstract algorithm types
@public AbstractNonlinearAlgorithm, AbstractDAEAlgorithm, AbstractSDEAlgorithm,
    AbstractLinearAlgorithm, AbstractRODEAlgorithm, AbstractSteadyStateAlgorithm,
    BasicEnsembleAlgorithm, EnsembleAlgorithm

# Abstract integrator types
@public DEIntegrator, AbstractODEIntegrator, AbstractSDEIntegrator,
    AbstractRODEIntegrator, AbstractDDEIntegrator, AbstractDAEIntegrator

# Abstract solution types
@public AbstractNoiseProcess, AbstractEnsembleSolution, AbstractNoTimeSolution,
    AbstractRODESolution

# Abstract function types
@public AbstractDiffEqFunction, AbstractODEFunction, AbstractSciMLFunction,
    AbstractParameterizedFunction

# Algorithm / problem traits
@public isadaptive, allowscomplex, allows_arbitrary_number_types, isautodifferentiable,
    is_diagonal_noise, forwarddiffs_model, forwarddiffs_model_time,
    allows_late_binding_tstops, alg_order, allowsbounds, allowsconstraints

# Initialization algorithms and interface
@public NoInit, OverrideInit, get_initial_values

# Solution / integrator interface
@public DEStats, check_error!, promote_tspan

# Problem types and alias specifiers
@public ImmutableODEProblem, NonlinearAliasSpecifier

# Steady-state / problem support types
@public AbstractSteadyStateProblem, StandardODEProblem

# Abstract solution / discretization types
@public AbstractTimeseriesSolution, AbstractDiscretization

# Interpolation types
@public AbstractDiffEqInterpolation, ConstantInterpolation, LinearInterpolation,
    HermiteInterpolation, SensitivityInterpolation

# Interpolation / symbolic / solution interface
@public interp_summary, getindepsym, getindepsym_defaultt,
    calculate_solution_errors!, initialize_dae!

# Automatic differentiation markers
@public NoAD

# Function-wrapper / solution interface helpers
@public unwrapped_f, solution_new_retcode

# Low-level solver-author extension entry points
@public __solve, __init

# Round 5: DDE/SDE/RODE/BVP solver-author extension API
# Abstract problem types solver packages subtype/dispatch on
@public AbstractBVProblem, AbstractRODEProblem, AbstractSDDEProblem

# Abstract algorithm / initialization-algorithm supertypes
@public AbstractDDEAlgorithm, DAEInitializationAlgorithm

# Abstract function types
@public AbstractDDEFunction, AbstractSDEFunction, AbstractSDDEFunction

# History function interface for delay equations
@public AbstractHistoryFunction

# Abstract integrator / solution types
@public AbstractSDDEIntegrator, AbstractODESolution

# Abstract solver cache supertype
@public DECache

# Specialization markers
@public FullSpecialize, NoSpecialize, FunctionWrapperSpecialize

# SDE interpretation trait
@public AlgorithmInterpretation, alg_interpretation

# AD / sensitivity function wrappers
@public TimeDerivativeWrapper, TimeGradientWrapper, UDerivativeWrapper, UJacobianWrapper

# Problem alias specifiers
@public RODEAliasSpecifier, SDEAliasSpecifier

end
