using SciMLBase, Test

if isdefined(Base, :ispublic)
    @testset "Clocks manual public API" begin
        for name in (
                :AbstractClock,
                :ContinuousClock,
                :PeriodicClock,
                :SolverStepClock,
                :EventClock,
                :TimeDomain,
                :Continuous,
                :Clock,
                :isclock,
                :issolverstepclock,
                :iscontinuous,
                :iseventclock,
                :is_discrete_time_domain,
                :first_clock_tick_time,
                :IndexedClock,
                :canonicalize_indexed_clock,
            )
            @test Base.ispublic(SciMLBase, name)
        end
    end

    @testset "SciMLFunctions manual public API" begin
        for name in (
                :AbstractSciMLFunction,
                :AbstractDiffEqFunction,
                :AbstractODEFunction,
                :AbstractSDEFunction,
                :AbstractDDEFunction,
                :AbstractDAEFunction,
                :AbstractRODEFunction,
                :AbstractDiscreteFunction,
                :AbstractSDDEFunction,
                :AbstractNonlinearFunction,
                :AbstractIntervalNonlinearFunction,
                :AbstractIntegralFunction,
                :AbstractOptimizationFunction,
                :AbstractODEInputFunction,
                :AbstractBVPFunction,
                :AbstractParameterizedFunction,
                :AbstractHistoryFunction,
            )
            @test Base.ispublic(SciMLBase, name)
        end
    end

    @testset "Solutions manual public API" begin
        for name in (
                :AbstractSciMLSolution,
                :AbstractNoTimeSolution,
                :AbstractTimeseriesSolution,
                :AbstractNoiseProcess,
                :AbstractEnsembleSolution,
                :AbstractLinearSolution,
                :AbstractEigenvalueSolution,
                :AbstractNonlinearSolution,
                :AbstractIntegralSolution,
                :AbstractOptimizationSolution,
                :AbstractSteadyStateSolution,
                :AbstractAnalyticalSolution,
                :AbstractODESolution,
                :AbstractDDESolution,
                :AbstractRODESolution,
                :AbstractDAESolution,
                :AbstractPDETimeSeriesSolution,
                :AbstractPDENoTimeSolution,
                :AbstractPDESolution,
                :AbstractSensitivitySolution,
            )
            @test Base.ispublic(SciMLBase, name)
        end
    end

    @testset "Problems manual public API" begin
        for name in (
                :AbstractSpecialization,
                :AbstractLinearProblem,
                :AbstractEigenvalueProblem,
                :AbstractIntervalNonlinearProblem,
                :AbstractIntegralProblem,
                :AbstractOptimizationProblem,
                :AbstractDynamicOptProblem,
                :AbstractAnalyticalProblem,
                :AbstractConstantLagDDEProblem,
                :AbstractSecondOrderODEProblem,
                :AbstractConstantLagSDDEProblem,
                :AbstractPDEProblem,
                :AbstractOptimizationCache,
            )
            @test Base.ispublic(SciMLBase, name)
        end
    end

    @testset "Algorithms manual public API" begin
        for name in (
                :isautodifferentiable,
                :allows_arbitrary_number_types,
                :allowscomplex,
                :isadaptive,
                :isdiscrete,
                :forwarddiffs_model,
                :forwarddiffs_model_time,
                :forwarddiff_chunksize,
                :has_lazy_interpolation,
                :allows_late_binding_tstops,
                :supports_opt_cache_interface,
                :has_init,
                :has_step,
                :alg_order,
                :allowsbounds,
                :requiresbounds,
                :allowsconstraints,
                :requiresconstraints,
                :requiresgradient,
                :allowsfg,
                :requireshessian,
                :allowsfgh,
                :requiresconsjac,
                :allowsconsjvp,
                :allowsconsvjp,
                :requiresconshess,
                :requireslagh,
                :allowscallback,
                :allows_non_wiener_noise,
                :requires_additive_noise,
                :AlgorithmInterpretation,
                :alg_interpretation,
                :AbstractSciMLAlgorithm,
                :AbstractDEAlgorithm,
                :AbstractLinearAlgorithm,
                :AbstractNonlinearAlgorithm,
                :AbstractIntervalNonlinearAlgorithm,
                :AbstractIntegralAlgorithm,
                :AbstractOptimizationAlgorithm,
                :AbstractSteadyStateAlgorithm,
                :AbstractBVPAlgorithm,
                :AbstractODEAlgorithm,
                :AbstractSecondOrderODEAlgorithm,
                :AbstractRODEAlgorithm,
                :AbstractSDEAlgorithm,
                :AbstractDAEAlgorithm,
                :AbstractDDEAlgorithm,
                :AbstractSDDEAlgorithm,
                :EnsembleAlgorithm,
                :DAEInitializationAlgorithm,
                :AbstractDiscretization,
                :AbstractDiscretizationMetadata,
                :NoInit,
                :CheckInit,
                :OverrideInit,
            )
            @test Base.ispublic(SciMLBase, name)
        end
    end

    @testset "Init-solve manual public API" begin
        for name in (
                :AbstractSteadyStateIntegrator,
                :AbstractSecondOrderODEIntegrator,
            )
            @test Base.ispublic(SciMLBase, name)
        end
    end

    @testset "Callbacks manual public API" begin
        for name in (
                :DECallback,
                :AbstractContinuousCallback,
                :AbstractDiscreteCallback,
                :ContinuousCallback,
                :VectorContinuousCallback,
                :DiscreteCallback,
                :CallbackSet,
                :NoRootFind,
                :LeftRootFind,
                :RightRootFind,
                :split_callbacks,
                :save_final_discretes!,
                :save_discretes_if_enabled!,
            )
            @test Base.ispublic(SciMLBase, name)
        end
    end

    @testset "Ensembles manual public API" begin
        for name in (
                :AbstractEnsembleProblem,
                :EnsembleProblem,
                :WeightedEnsembleProblem,
                :DEFAULT_PROB_FUNC,
                :DEFAULT_OUTPUT_FUNC,
                :DEFAULT_REDUCTION,
                :__solve,
                :EnsembleContext,
                :generate_sim_seeds,
                :default_rng_func,
                :BasicEnsembleAlgorithm,
                :EnsembleSerial,
                :EnsembleThreads,
                :EnsembleDistributed,
                :EnsembleSplitThreads,
                :AbstractEnsembleEstimator,
                :EnsembleSolution,
                :EnsembleTestSolution,
                :WeightedEnsembleSolution,
            )
            @test Base.ispublic(SciMLBase, name)
        end
    end
end
