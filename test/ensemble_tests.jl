using Test, SciMLBase, SciMLBase.EnsembleAnalysis
using SciMLBase: RecipesBase
using Logging

EA = SciMLBase.EnsembleAnalysis

@testset "ensemble progress aggregation" begin
    io = IOBuffer()
    logger = ConsoleLogger(io, LogLevel(-1))
    state = SciMLBase.AggregateProgressState(
        Dict{Symbol, Float64}(), 0, 0.0, 0.0, ReentrantLock(), logger
    )
    args = (
        ; level = LogLevel(-1), message = "trajectory", _module = Main,
        group = :test, id = :trajectory, file = "test", line = 1,
        kwargs = (; progress = 0.5),
    )

    @test !SciMLBase._aggregate_progress_filter!(state, args)
    @test occursin("Total", String(take!(io)))
end

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

# https://github.com/SciML/DifferentialEquations.jl/issues/632
@testset "EnsembleSummary rejects complex-valued trajectories" begin
    prob = ODEProblem((u, p, t) -> u, 1.0 + 0.0im, (0.0, 1.0))
    ts = [0.0, 0.5, 1.0]
    vals = [
        [1.0 + 0.0im, 1.5 + 0.2im, 2.0 + 0.4im],
        [0.5 + 0.1im, 1.0 + 0.3im, 1.5 + 0.5im],
        [1.2 - 0.1im, 1.4 + 0.0im, 1.8 + 0.2im],
    ]
    sols = [
        SciMLBase.build_solution(
            prob, :NoAlgorithm, ts, u; retcode = SciMLBase.ReturnCode.Success
        )
            for u in vals
    ]
    sim = EnsembleSolution(sols, 0.0, true)
    err = try
        EnsembleSummary(sim, ts)
        nothing
    catch e
        e
    end
    @test err isa SciMLBase.ComplexEnsembleSummaryError
    @test occursin("Complex-valued ensemble", sprint(showerror, err))
    # mean/var path remains available without ordering
    m, v = EA.timeseries_point_meanvar(sim, ts)
    @test length(m.u) == length(ts)
end

# https://github.com/SciML/SciMLBase.jl/issues/1654
# Plots normally defines this; the stub lets the recipe run without a backend
RecipesBase.is_key_supported(::Symbol) = true
@testset "EnsembleSummary SEM ribbon for scalar states" begin
    function sem_ribbon(summ)
        series = RecipesBase.apply_recipe(Dict{Symbol, Any}(:ci_type => :SEM), summ)
        return series[1].plotattributes[:ribbon][1]
    end

    ts = [0.0, 0.5, 1.0]
    vals = [[1.0, 2.0, 4.0], [2.0, 3.0, 5.0], [4.0, 7.0, 11.0], [0.5, 1.5, 3.5]]
    N = length(vals)
    prob_s = ODEProblem((u, p, t) -> u, 0.0, (0.0, 1.0))
    prob_v = ODEProblem((u, p, t) -> u, [0.0], (0.0, 1.0))
    build(prob, u) = SciMLBase.build_solution(
        prob, :NoAlgorithm, ts, u; retcode = SciMLBase.ReturnCode.Success
    )
    summ_s = EnsembleSummary(EnsembleSolution([build(prob_s, u) for u in vals], 0.0, true))
    summ_v = EnsembleSummary(
        EnsembleSolution([build(prob_v, [[x] for x in u]) for u in vals], 0.0, true)
    )

    @test summ_s.num_monte == N
    ribbon_s = sem_ribbon(summ_s)
    @test ribbon_s ≈ 1.96 .* sqrt.(summ_s.v.u ./ N)
    @test ribbon_s ≈ sem_ribbon(summ_v)
end
