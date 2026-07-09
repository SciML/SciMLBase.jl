using SciMLBase, Test

if isdefined(Base, :ispublic)
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
                :AbstractIntervalNonlinearAlgorithm,
                :AbstractOptimizationAlgorithm,
                :AbstractBVPAlgorithm,
                :AbstractSecondOrderODEAlgorithm,
                :CheckInit,
            )
            @test Base.ispublic(SciMLBase, name)
        end
    end
end
