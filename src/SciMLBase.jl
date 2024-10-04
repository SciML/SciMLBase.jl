module SciMLBase
if isdefined(Base, :Experimental) &&
   isdefined(Base.Experimental, Symbol("@max_methods"))
    @eval Base.Experimental.@max_methods 1
end
using ConstructionBase
using RecipesBase, RecursiveArrayTools
using SciMLStructures
using SymbolicIndexingInterface
using DocStringExtensions
using LinearAlgebra
using Statistics
using Distributed
using Markdown
using Printf
import Preferences

import Logging, ArrayInterface
import IteratorInterfaceExtensions
import CommonSolve: solve, init, step!, solve!
import FunctionWrappersWrappers
import RuntimeGeneratedFunctions
import EnumX
import ADTypes: AbstractADType
import Accessors: @set, @reset
using Expronicon.ADT: @match

using Reexport
using SciMLOperators
using SciMLOperators:
                      AbstractSciMLOperator,
                      IdentityOperator, NullOperator,
                      ScaledOperator, AddedOperator, ComposedOperator,
                      InvertedOperator, InvertibleOperator

import SciMLOperators:
                       DEFAULT_UPDATE_FUNC, update_coefficients, update_coefficients!,
                       getops, isconstant, iscached, islinear, issquare,
                       has_adjoint, has_expmv, has_expmv!, has_exp,
                       has_mul, has_mul!, has_ldiv, has_ldiv!

@reexport using SciMLOperators

function __solve end
function __init end

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
abstract type AbstractNonlinearProblem{uType, isinplace} <: AbstractDEProblem end
abstract type AbstractIntervalNonlinearProblem{uType, isinplace} <:
              AbstractNonlinearProblem{uType,
    isinplace} end
const AbstractSteadyStateProblem{uType, isinplace} = AbstractNonlinearProblem{uType,
    isinplace}

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
$(TYPEDEF)
"""
struct NoInit <: DAEInitializationAlgorithm end

"""
$(TYPEDEF)
"""
struct CheckInit <: DAEInitializationAlgorithm end

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
"""
abstract type AbstractNoTimeSolution{T, N} <: AbstractArray{T, N} end

"""
$(TYPEDEF)
"""
abstract type AbstractTimeseriesSolution{T, N, A} <: AbstractDiffEqArray{T, N, A} end

"""
$(TYPEDEF)
"""
abstract type AbstractEnsembleSolution{T, N, A} <: AbstractVectorOfArray{T, N, A} end

"""
$(TYPEDEF)
"""
abstract type AbstractNoiseProcess{T, N, A, isinplace} <: AbstractDiffEqArray{T, N, A} end

"""
Union of all base solution types.

Uses a Union so that solution types can be `<: AbstractArray`
"""
const AbstractSciMLSolution = Union{AbstractTimeseriesSolution,
    AbstractNoTimeSolution,
    AbstractEnsembleSolution,
    AbstractNoiseProcess}

"""
$(TYPEDEF)
"""
abstract type AbstractLinearSolution{T, N} <: AbstractNoTimeSolution{T, N} end

"""
$(TYPEDEF)
"""
abstract type AbstractNonlinearSolution{T, N} <: AbstractNoTimeSolution{T, N} end

"""
$(TYPEDEF)
"""
abstract type AbstractIntegralSolution{T, N} <: AbstractNoTimeSolution{T, N} end

"""
$(TYPEDEF)
"""
abstract type AbstractOptimizationSolution{T, N} <: AbstractNoTimeSolution{T, N} end

"""
$(TYPEDEF)
"""
const AbstractSteadyStateSolution{T, N} = AbstractNonlinearSolution{T, N}

"""
$(TYPEDEF)
"""
abstract type AbstractAnalyticalSolution{T, N, S} <: AbstractTimeseriesSolution{T, N, S} end

"""
$(TYPEDEF)
"""
abstract type AbstractODESolution{T, N, S} <: AbstractTimeseriesSolution{T, N, S} end

# Needed for plot recipes
"""
$(TYPEDEF)
"""
abstract type AbstractDDESolution{T, N, S} <: AbstractODESolution{T, N, S} end

"""
$(TYPEDEF)
"""
abstract type AbstractRODESolution{T, N, S} <: AbstractODESolution{T, N, S} end

"""
$(TYPEDEF)
"""
abstract type AbstractDAESolution{T, N, S} <: AbstractODESolution{T, N, S} end

"""
$(TYPEDEF)
"""
abstract type AbstractPDETimeSeriesSolution{T, N, S, D} <:
              AbstractTimeseriesSolution{T, N, S} end

"""
$(TYPEDEF)
"""
abstract type AbstractPDENoTimeSolution{T, N, S, D} <:
              AbstractNoTimeSolution{T, N} end

"""
$(TYPEDEF)
"""
const AbstractPDESolution{T, N, S, D} = Union{AbstractPDETimeSeriesSolution{T, N, S, D},
    AbstractPDENoTimeSolution{T, N, S, D}}

"""
$(TYPEDEF)
"""
abstract type AbstractSensitivitySolution{T, N, S} <: AbstractTimeseriesSolution{T, N, S} end

# Misc
# TODO - deprecate AbstractDiffEqOperator family
"""
$(TYPEDEF)
"""
abstract type AbstractDiffEqOperator{T} <: AbstractSciMLOperator{T} end

"""
$(TYPEDEF)
"""
abstract type AbstractDiffEqLinearOperator{T} <: AbstractDiffEqOperator{T} end

"""
$(TYPEDEF)
"""
abstract type AbstractDiffEqCompositeOperator{T} <: AbstractDiffEqLinearOperator{T} end

"""
$(TYPEDEF)

Base for types defining SciML functions.
"""
abstract type AbstractSciMLFunction{iip} <: Function end

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
abstract type AbstractReactionNetwork <: Function end

"""
$(TYPEDEF)

Internal. Used for signifying which AD context a derivative calculation is in.
"""
abstract type ADOriginator end

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

include("utils.jl")
include("function_wrappers.jl")
include("scimlfunctions.jl")
include("alg_traits.jl")
include("debug.jl")

unwrapped_f(f) = f
unwrapped_f(f::Void) = unwrapped_f(f.f)
function unwrapped_f(f::FunctionWrappersWrappers.FunctionWrappersWrapper)
    unwrapped_f(f.fw[1].obj[])
end

function specialization(::Union{ODEFunction{iip, specialize},
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
        BatchIntegralFunction{iip, specialize}}) where {iip,
        specialize}
    specialize
end

specialization(f::AbstractSciMLFunction) = FullSpecialize

"""
$(TYPEDEF)
"""
abstract type AbstractParameterizedFunction{iip} <: AbstractODEFunction{iip} end

include("retcodes.jl")
include("operators/operators.jl")
include("operators/basic_operators.jl")
include("operators/diffeq_operator.jl")
include("operators/common_defaults.jl")
include("symbolic_utils.jl")
include("performance_warnings.jl")

include("problems/discrete_problems.jl")
include("problems/implicit_discrete_problems.jl")
include("problems/steady_state_problems.jl")
include("problems/analytical_problems.jl")
include("problems/linear_problems.jl")
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

include("deprecated.jl")

import PrecompileTools

PrecompileTools.@compile_workload begin
    function lorenz(du, u, p, t)
        du[1] = 10.0(u[2] - u[1])
        du[2] = u[1] * (28.0 - u[3]) - u[2]
        du[3] = u[1] * u[2] - (8 / 3) * u[3]
    end

    function lorenz_oop(u, p, t)
        [10.0(u[2] - u[1]), u[1] * (28.0 - u[3]) - u[2], u[1] * u[2] - (8 / 3) * u[3]]
    end

    ODEProblem(lorenz, [1.0; 0.0; 0.0], (0.0, 1.0))
    ODEProblem(lorenz, [1.0; 0.0; 0.0], (0.0, 1.0), Float64[])
    ODEProblem(lorenz_oop, [1.0; 0.0; 0.0], (0.0, 1.0))
    ODEProblem(lorenz_oop, [1.0; 0.0; 0.0], (0.0, 1.0), Float64[])
end

function discretize end
function symbolic_discretize end

isfunctionwrapper(x) = false
function wrapfun_oop end
function wrapfun_iip end
function unwrap_fw end

export ReturnCode

export DEAlgorithm, SciMLAlgorithm, DEProblem, DEAlgorithm, DESolution, SciMLSolution

# Exports
export AllObserved

export isinplace

export solve, solve!, init, discretize, symbolic_discretize

export LinearProblem, NonlinearProblem, IntervalNonlinearProblem,
       IntegralProblem, SampledIntegralProblem, OptimizationProblem,
       NonlinearLeastSquaresProblem

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
export PDEProblem
export IncrementingODEProblem

export BVProblem, TwoPointBVProblem, SecondOrderBVProblem, TwoPointSecondOrderBVProblem

export remake

export ODEFunction, DiscreteFunction, ImplicitDiscreteFunction, SplitFunction, DAEFunction,
       DDEFunction, SDEFunction, SplitSDEFunction, RODEFunction, SDDEFunction,
       IncrementingODEFunction, NonlinearFunction, IntervalNonlinearFunction, BVPFunction,
       DynamicalBVPFunction, IntegralFunction, BatchIntegralFunction

export OptimizationFunction, MultiObjectiveOptimizationFunction

export EnsembleThreads, EnsembleDistributed, EnsembleSplitThreads, EnsembleSerial

export EnsembleAnalysis, EnsembleSummary

export tuples, intervals, TimeChoiceIterator

export AffineDiffEqOperator, DiffEqScaledOperator

export DiffEqScalar, DiffEqArrayOperator, DiffEqIdentity

export step!, deleteat!, addat!, get_tmp_cache,
       full_cache, user_cache, u_cache, du_cache,
       rand_cache, ratenoise_cache,
       resize_non_user_cache!, deleteat_non_user_cache!, addat_non_user_cache!,
       terminate!,
       add_tstop!, has_tstop, first_tstop, pop_tstop!,
       add_saveat!, set_abstol!,
       set_reltol!, get_du, get_du!, get_dt, get_proposed_dt, set_proposed_dt!,
       u_modified!, savevalues!, reinit!, auto_dt_reset!, set_t!,
       set_u!, check_error, change_t_via_interpolation!, addsteps!,
       isdiscrete, reeval_internals_due_to_modification!

export ContinuousCallback, DiscreteCallback, CallbackSet, VectorContinuousCallback

export Clocks, TimeDomain, is_discrete_time_domain, isclock, issolverstepclock, iscontinuous

end
