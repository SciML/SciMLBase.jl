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
                :AbstractSciMLAlgorithm,
                :AbstractIntervalNonlinearAlgorithm,
                :AbstractOptimizationAlgorithm,
                :AbstractBVPAlgorithm,
                :AbstractSecondOrderODEAlgorithm,
                :AbstractSDDEAlgorithm,
                :CheckInit,
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
end
