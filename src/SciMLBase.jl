module SciMLBase

using ConstructionBase
using RecipesBase, RecursiveArrayTools, Tables, TreeViews
using DocStringExtensions
using LinearAlgebra
using Statistics
using Distributed
using StaticArrays
using Markdown

import Logging, ArrayInterfaceCore
import IteratorInterfaceExtensions
import CommonSolve: solve, init, solve!

function __solve end
function __init end

"""
$(TYPEDEF)
"""
abstract type SciMLProblem end

# Problems
"""
$(TYPEDEF)

Base type for all DifferentialEquations.jl problems. Concrete subtypes of
`DEProblem` contain the necessary information to fully define a differential
equation of the corresponding type.
"""
abstract type DEProblem <: SciMLProblem end

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
abstract type AbstractLinearProblem{bType, isinplace} <: SciMLProblem end

"""
$(TYPEDEF)

Base for types which define integrals suitable for quadrature.
"""
abstract type AbstractIntegralProblem{isinplace} <: SciMLProblem end

"""
$(TYPEDEF)

Base for types which define equations for optimization.
"""
abstract type AbstractOptimizationProblem{isinplace} <: SciMLProblem end

"""
$(TYPEDEF)

Base for types which define nonlinear solve problems (f(u)=0).
"""
abstract type AbstractNonlinearProblem{uType, isinplace} <: DEProblem end
const AbstractSteadyStateProblem{uType, isinplace} = AbstractNonlinearProblem{uType,
                                                                              isinplace}

"""
$(TYPEDEF)
"""
abstract type AbstractNoiseProblem <: DEProblem end

"""
$(TYPEDEF)

Base for types which define ODE problems.
"""
abstract type AbstractODEProblem{uType, tType, isinplace} <: DEProblem end

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
abstract type AbstractRODEProblem{uType, tType, isinplace, ND} <: DEProblem end

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
abstract type AbstractDAEProblem{uType, duType, tType, isinplace} <: DEProblem end

"""
$(TYPEDEF)

Base for types which define DDE problems.
"""
abstract type AbstractDDEProblem{uType, tType, lType, isinplace} <: DEProblem end

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
abstract type AbstractBVProblem{uType, tType, isinplace} <:
              AbstractODEProblem{uType, tType, isinplace} end

"""
$(TYPEDEF)

Base for types which define jump problems.
"""
abstract type AbstractJumpProblem{P, J} <: DEProblem end

"""
$(TYPEDEF)

Base for types which define SDDE problems.
"""
abstract type AbstractSDDEProblem{uType, tType, lType, isinplace, ND} <: DEProblem end

"""
$(TYPEDEF)
"""
abstract type AbstractConstantLagSDDEProblem{uType, tType, lType, isinplace, ND} <:
              AbstractSDDEProblem{uType, tType, lType, isinplace, ND} end

"""
$(TYPEDEF)

Base for types which define PDE problems.
"""
abstract type AbstractPDEProblem <: DEProblem end

# Algorithms
"""
$(TYPEDEF)
"""
abstract type SciMLAlgorithm end

"""
$(TYPEDEF)
"""
abstract type DEAlgorithm <: SciMLAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractLinearAlgorithm <: SciMLAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractNonlinearAlgorithm <: SciMLAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractIntegralAlgorithm <: SciMLAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractOptimizationAlgorithm <: DEAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractSteadyStateAlgorithm <: DEAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractODEAlgorithm <: DEAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractSecondOrderODEAlgorithm <: DEAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractRODEAlgorithm <: DEAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractSDEAlgorithm <: DEAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractDAEAlgorithm <: DEAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractDDEAlgorithm <: DEAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractSDDEAlgorithm <: DEAlgorithm end

"""
$(TYPEDEF)
"""
abstract type EnsembleAlgorithm <: SciMLAlgorithm end

"""
$(TYPEDEF)
"""
abstract type AbstractSensitivityAlgorithm{CS, AD, FDT} <: SciMLAlgorithm end

"""
$(TYPEDEF)
"""
abstract type DAEInitializationAlgorithm <: SciMLAlgorithm end

"""
$(TYPEDEF)
"""
struct NoInit <: DAEInitializationAlgorithm end

# PDE Discretizations

"""
$(TYPEDEF)
"""
abstract type AbstractDiscretization <: SciMLAlgorithm end

# Monte Carlo Simulations
"""
$(TYPEDEF)
"""
abstract type AbstractEnsembleProblem <: SciMLProblem end

"""
$(TYPEDEF)
"""
abstract type AbstractEnsembleEstimator <: SciMLAlgorithm end

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
const SciMLSolution = Union{AbstractTimeseriesSolution,
                            AbstractNoTimeSolution,
                            AbstractEnsembleSolution,
                            AbstractNoiseProcess}
const DESolution = SciMLSolution
export SciMLSolution, DESolution

export AllObserved

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
abstract type AbstractSensitivitySolution{T, N, S} <: AbstractTimeseriesSolution{T, N, S} end

# Misc
"""
$(TYPEDEF)
"""
abstract type AbstractSciMLOperator{T} end

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
abstract type AbstractDiffEqFunction{iip} <: AbstractSciMLFunction{iip} end

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

Base type for AD choices.
"""
abstract type AbstractADType end

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

include("operators/operators.jl")
include("operators/basic_operators.jl")
include("operators/diffeq_operator.jl")
include("operators/common_defaults.jl")

include("problems/problem_utils.jl")
include("problems/discrete_problems.jl")
include("problems/steady_state_problems.jl")
include("problems/analytical_problems.jl")
include("problems/basic_problems.jl")
include("problems/ode_problems.jl")
include("problems/rode_problems.jl")
include("problems/sde_problems.jl")
include("problems/noise_problems.jl")
include("problems/bvp_problems.jl")
include("problems/dae_problems.jl")
include("problems/dde_problems.jl")
include("problems/sdde_problems.jl")
include("problems/pde_problems.jl")
include("problems/problem_traits.jl")

include("solutions/basic_solutions.jl")
include("solutions/nonlinear_solutions.jl")
include("solutions/ode_solutions.jl")
include("solutions/rode_solutions.jl")
include("solutions/optimization_solutions.jl")
include("solutions/dae_solutions.jl")
include("solutions/solution_interface.jl")

include("ensemble/ensemble_solutions.jl")
include("ensemble/ensemble_problems.jl")
include("ensemble/basic_ensemble_solve.jl")
include("ensemble/ensemble_analysis.jl")

include("solve.jl")
include("interpolation.jl")
include("integrator_interface.jl")
include("tabletraits.jl")
include("remake.jl")
include("callbacks.jl")

function discretize end
function symbolic_discretize end

isfunctionwrapper(x) = false
function wrapfun_oop end
function wrapfun_iip end
function unwrap_fw end

# Deprecated Quadrature things
const AbstractQuadratureProblem = AbstractIntegralProblem
const AbstractQuadratureAlgorithm = AbstractIntegralAlgorithm
const AbstractQuadratureSolution = AbstractIntegralSolution

export isinplace

export solve, solve!, init, discretize, symbolic_discretize

export LinearProblem, NonlinearProblem, IntegralProblem, OptimizationProblem

# Deprecated
export IntegralProblem

export DiscreteProblem
export SteadyStateProblem, SteadyStateSolution
export NoiseProblem
export ODEProblem, ODESolution
export DynamicalODEFunction, DynamicalODEProblem,
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

export BVProblem, TwoPointBVProblem

export remake

export ODEFunction, DiscreteFunction, SplitFunction, DAEFunction, DDEFunction,
       SDEFunction, SplitSDEFunction, RODEFunction, SDDEFunction, IncrementingODEFunction,
       NonlinearFunction

export OptimizationFunction

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

export update_coefficients!, update_coefficients,
       has_adjoint, has_expmv!, has_expmv, has_exp, has_mul, has_mul!, has_ldiv, has_ldiv!

export ContinuousCallback, DiscreteCallback, CallbackSet, VectorContinuousCallback

end
