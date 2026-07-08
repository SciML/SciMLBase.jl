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
common implementation across all solvers. See also [`__init`](@ref).
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
solvers. See also [`__solve`](@ref).
"""
function __init end

# Local alias for the `Union{Function, Type}` callable bound (mirrors the
# unexported `Base.Callable`), so problem constructors can dispatch on it
# without a non-public qualified access to Base.
const Callable = Union{Function, Type}

"""
$(TYPEDEF)
"""
abstract type AbstractSciMLProblem end

# Problems
"""
$(TYPEDEF)

Base type for all DifferentialEquations.jl problems. Concrete subtypes of
`AbstractDEProblem` contain the necessary information to fully define a differential
equation of the corresponding type.
"""
abstract type AbstractDEProblem <: AbstractSciMLProblem end

"""
$(TYPEDEF)
"""
abstract type DEElement end

"""
$(TYPEDEF)
"""
abstract type DESensitivity end

"""
$(TYPEDEF)

Base for types which define linear systems.
"""
abstract type AbstractLinearProblem{bType, isinplace} <: AbstractSciMLProblem end

"""
$(TYPEDEF)

Base for types which define eigenvalue problems.
"""
abstract type AbstractEigenvalueProblem <: AbstractSciMLProblem end

"""
$(TYPEDEF)

Base for types which define integrals suitable for quadrature.
"""
abstract type AbstractIntegralProblem{isinplace} <: AbstractSciMLProblem end

"""
$(TYPEDEF)

Base for types which define equations for optimization.
"""
abstract type AbstractOptimizationProblem{isinplace} <: AbstractSciMLProblem end

"""
$(TYPEDEF)

Base for types which define caches for optimization problems. Must at least hold the optimization
function `f <: OptimizationFunction` and the parameters `p`.
"""
abstract type AbstractOptimizationCache end

"""
$(TYPEDEF)

Base for types which define nonlinear solve problems (`f(u)=0`).
"""
abstract type AbstractNonlinearProblem{uType, isinplace} <: AbstractSciMLProblem end
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
"""
abstract type AbstractNoiseProblem <: AbstractDEProblem end

"""
$(TYPEDEF)

Base for types which define ODE problems.
"""
abstract type AbstractODEProblem{uType, tType, isinplace} <: AbstractDEProblem end

"""
$(TYPEDEF)

Base for types which define dynamical optimization problems.
"""
abstract type AbstractDynamicOptProblem{uType, tType, isinplace} <:
AbstractODEProblem{uType, tType, isinplace} end

"""
$(TYPEDEF)

Base for types which define discrete problems.
"""
abstract type AbstractDiscreteProblem{uType, tType, isinplace} <:
AbstractODEProblem{uType, tType, isinplace} end

"""
$(TYPEDEF)
"""
abstract type AbstractAnalyticalProblem{uType, tType, isinplace} <:
AbstractODEProblem{uType, tType, isinplace} end

"""
$(TYPEDEF)

Base for types which define RODE problems.
"""
abstract type AbstractRODEProblem{uType, tType, isinplace, ND} <: AbstractDEProblem end

"""
$(TYPEDEF)

Base for types which define SDE problems.
"""
abstract type AbstractSDEProblem{uType, tType, isinplace, ND} <:
AbstractRODEProblem{uType, tType, isinplace, ND} end

"""
$(TYPEDEF)

Base for types which define DAE problems.
"""
abstract type AbstractDAEProblem{uType, duType, tType, isinplace} <: AbstractDEProblem end

"""
$(TYPEDEF)

Base for types which define DDE problems.
"""
abstract type AbstractDDEProblem{uType, tType, lType, isinplace} <: AbstractDEProblem end

"""
$(TYPEDEF)
"""
abstract type AbstractConstantLagDDEProblem{uType, tType, lType, isinplace} <:
AbstractDDEProblem{uType, tType, lType, isinplace} end

"""
$(TYPEDEF)
"""
abstract type AbstractSecondOrderODEProblem{uType, tType, isinplace} <:
AbstractODEProblem{uType, tType, isinplace} end

"""
$(TYPEDEF)

Base for types which define BVP problems.
"""
abstract type AbstractBVProblem{uType, tType, isinplace, nlls} <:
AbstractODEProblem{uType, tType, isinplace} end

"""
$(TYPEDEF)

Base for types which define jump problems.
"""
abstract type AbstractJumpProblem{P, J} <: AbstractDEProblem end

"""
$(TYPEDEF)

Base for types which define SDDE problems.
"""
abstract type AbstractSDDEProblem{uType, tType, lType, isinplace, ND} <: AbstractDEProblem end

"""
$(TYPEDEF)
"""
abstract type AbstractConstantLagSDDEProblem{uType, tType, lType, isinplace, ND} <:
AbstractSDDEProblem{uType, tType, lType, isinplace, ND} end

"""
$(TYPEDEF)

Base for types which define PDE problems.
"""
abstract type AbstractPDEProblem <: AbstractDEProblem end

# Algorithms
"""
$(TYPEDEF)
"""
abstract type AbstractSciMLAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractDEAlgorithm <: AbstractSciMLAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractLinearAlgorithm <: AbstractSciMLAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractNonlinearAlgorithm <: AbstractSciMLAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractIntervalNonlinearAlgorithm <: AbstractSciMLAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractIntegralAlgorithm <: AbstractSciMLAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractOptimizationAlgorithm <: AbstractDEAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractSteadyStateAlgorithm <: AbstractDEAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractBVPAlgorithm <: AbstractDEAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractODEAlgorithm <: AbstractDEAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractSecondOrderODEAlgorithm <: AbstractDEAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractRODEAlgorithm <: AbstractDEAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractSDEAlgorithm <: AbstractDEAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractDAEAlgorithm <: AbstractDEAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractDDEAlgorithm <: AbstractDEAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractSDDEAlgorithm <: AbstractDEAlgorithm end

"""
$(TYPEDEF)

The supertype of all ensemble algorithms, i.e. the algorithms passed as the
`ensemblealg` argument to `solve(::EnsembleProblem, alg, ensemblealg; ...)` to control
how the many trajectories of an [`EnsembleProblem`](@ref) are executed. Subtypes select
the parallelization strategy (serial, threaded, distributed, GPU, ...); see
[`BasicEnsembleAlgorithm`](@ref) for the CPU-parallelism implementations built into
SciMLBase.
"""
abstract type EnsembleAlgorithm <: AbstractSciMLAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractSensitivityAlgorithm{CS, AD, FDT} <: AbstractSciMLAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractOverloadingSensitivityAlgorithm{CS, AD, FDT} <:
AbstractSensitivityAlgorithm{CS, AD, FDT} end

"""
$(TYPEDEF)
"""
abstract type AbstractForwardSensitivityAlgorithm{CS, AD, FDT} <:
AbstractOverloadingSensitivityAlgorithm{CS, AD, FDT} end

"""
$(TYPEDEF)
"""
abstract type AbstractAdjointSensitivityAlgorithm{CS, AD, FDT} <:
AbstractOverloadingSensitivityAlgorithm{CS, AD, FDT} end

"""
$(TYPEDEF)
"""
abstract type AbstractSecondOrderSensitivityAlgorithm{CS, AD, FDT} <:
AbstractOverloadingSensitivityAlgorithm{CS, AD, FDT} end

"""
$(TYPEDEF)
"""
abstract type AbstractShadowingSensitivityAlgorithm{CS, AD, FDT} <:
AbstractOverloadingSensitivityAlgorithm{CS, AD, FDT} end

"""
$(TYPEDEF)
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
"""
abstract type AbstractDiscretization <: AbstractSciMLAlgorithm end

# Discretization metadata
"""
$(TYPEDEF)
"""
abstract type AbstractDiscretizationMetadata{hasTime} end

# Monte Carlo Simulations
"""
$(TYPEDEF)

The supertype of all ensemble problem types. An ensemble problem wraps a base
`AbstractSciMLProblem` together with the functions (`prob_func`, `output_func`,
`reduction`, ...) needed to generate, run, and reduce many related trajectories under
the [parallel ensemble interface](@ref ensemble). The concrete implementation is
[`EnsembleProblem`](@ref); solving one dispatches on the chosen
[`EnsembleAlgorithm`](@ref).
"""
abstract type AbstractEnsembleProblem <: AbstractSciMLProblem end

"""
$(TYPEDEF)
"""
abstract type AbstractEnsembleEstimator <: AbstractSciMLAlgorithm end

export EnsembleProblem
export EnsembleSolution, EnsembleTestSolution, EnsembleSummary

"""
$(TYPEDEF)
"""
abstract type AbstractDiffEqInterpolation end

"""
$(TYPEDEF)
"""
abstract type AbstractDEOptions end

"""
$(TYPEDEF)
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
"""
abstract type DEIntegrator{Alg, IIP, U, T} end

"""
$(TYPEDEF)
"""
abstract type AbstractSteadyStateIntegrator{Alg, IIP, U} <:
DEIntegrator{Alg, IIP, U, Nothing} end

"""
$(TYPEDEF)
"""
abstract type AbstractODEIntegrator{Alg, IIP, U, T} <: DEIntegrator{Alg, IIP, U, T} end

"""
$(TYPEDEF)
"""
abstract type AbstractSecondOrderODEIntegrator{Alg, IIP, U, T} <:
DEIntegrator{Alg, IIP, U, T} end

"""
$(TYPEDEF)
"""
abstract type AbstractRODEIntegrator{Alg, IIP, U, T} <: DEIntegrator{Alg, IIP, U, T} end

"""
$(TYPEDEF)
"""
abstract type AbstractSDEIntegrator{Alg, IIP, U, T} <: DEIntegrator{Alg, IIP, U, T} end

"""
$(TYPEDEF)
"""
abstract type AbstractDDEIntegrator{Alg, IIP, U, T} <: DEIntegrator{Alg, IIP, U, T} end

"""
$(TYPEDEF)
"""
abstract type AbstractDAEIntegrator{Alg, IIP, U, T} <: DEIntegrator{Alg, IIP, U, T} end

"""
$(TYPEDEF)
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

Base for types defining SciML functions.
"""
abstract type AbstractSciMLFunction{iip} end

"""
$(TYPEDEF)

Base for types defining differential equation functions.
"""
abstract type AbstractDiffEqFunction{iip} <:
AbstractSciMLFunction{iip} end

"""
$(TYPEDEF)

Base for types defining integrand functions.
"""
abstract type AbstractIntegralFunction{iip} <:
AbstractSciMLFunction{iip} end

"""
$(TYPEDEF)

Base for types defining optimization functions.
"""
abstract type AbstractOptimizationFunction{iip} <: AbstractSciMLFunction{iip} end

"""
$(TYPEDEF)

Base for types which define the history of a delay differential equation.
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

Used to specify which variables can be aliased in a solve.
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

Transform a symbolic or high-level PDE/system description into the numerical
object consumed by solvers, commonly an `AbstractSciMLProblem` or a lower-level
discretized system. Discretizer packages define methods for their concrete
system and discretizer types.
"""
function discretize end

"""
    symbolic_discretize(sys, discretizer, args...; kwargs...)

Return the symbolic representation of a discretization when the discretizer can
expose one. This is primarily used for inspecting or transforming the generated
system before numeric problem construction.
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
