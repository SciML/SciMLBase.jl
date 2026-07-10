using Test, SciMLBase, SciMLBase.EnsembleAnalysis

EA = SciMLBase.EnsembleAnalysis

# tests for https://github.com/SciML/DifferentialEquations.jl/issues/731
# make sure integer inputs work
@test all(EA.componentwise_mean([[1, 1], [2, 2], [3, 3]]) .≈ [2.0, 2.0])
m, v = EA.componentwise_meanvar([[1, 1], [2, 2], [3, 3]])
@test all(m .≈ 2.0)
@test all(v .≈ 1.0)
mx, my, C = EA.componentwise_meancov([[1, 1], [2, 2], [3, 3]], [[3, 3], [2, 2], [1, 1]])
@test all(mx .≈ 2.0)
@test all(my .≈ 2.0)
@test all(C .≈ -1.0)
mx, my,
    C = EA.componentwise_weighted_meancov(
    [[1, 1], [2, 2], [3, 3]],
    [[3, 3], [2, 2], [1, 1]],
    [[1, 1], [2, 2], [1, 1]]
)
@test all(mx .≈ 2.0)
@test all(my .≈ 2.0)
@test all(C .≈ -0.8)

@testset "weighted full-series covariance helpers" begin
    prob = ODEProblem((u, p, t) -> u, 0.0, (0.0, 2.0))
    ts = [0.0, 1.0, 2.0]
    vals = [[1.0, 2.0, 4.0], [2.0, 3.0, 5.0], [4.0, 7.0, 11.0]]
    sols = [
        SciMLBase.build_solution(
                prob, :NoAlgorithm, ts, u; retcode = SciMLBase.ReturnCode.Success
            )
            for u in vals
    ]
    sim = EnsembleSolution(sols, 0.0, true)
    weights = [0.2, 0.3, 0.5]

    step_meancovs = EA.timeseries_steps_meancov(sim)
    @test size(step_meancovs) == (length(ts), length(ts))
    @test all(isapprox.(step_meancovs[2, 3], EA.timestep_meancov(sim, 2, 3)))

    step_covs = EA.timeseries_steps_weighted_meancov(sim, weights)
    @test size(step_covs) == (length(ts), length(ts))
    @test all(isapprox.(step_covs[2, 3], EA.timestep_weighted_meancov(sim, weights, 2, 3)))

    point_meancovs = EA.timeseries_point_meancov(sim, ts[1:2], ts)
    @test size(point_meancovs) == (2, length(ts))
    @test all(isapprox.(point_meancovs[2, 3], EA.timepoint_meancov(sim, ts[2], ts[3])))

    point_covs = EA.timeseries_point_weighted_meancov(sim, weights, ts, ts)
    @test size(point_covs) == (length(ts), length(ts))
    @test all(
        isapprox.(
            point_covs[2, 3], EA.timepoint_weighted_meancov(sim, weights, ts[2], ts[3])
        )
    )
end
