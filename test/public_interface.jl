using SciMLBase, Test

if isdefined(Base, :ispublic)
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
