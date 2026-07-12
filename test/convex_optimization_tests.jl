using SciMLBase, Test
using SciMLBase: MinSense, MaxSense, build_convex_solution, DefaultOptimizationCache,
    NullParameters, ReturnCode, OptimizationFunction

@testset "ConvexOptimizationProblem construction" begin
    prob = ConvexOptimizationProblem((u, p) -> sum(abs2, u), [1.0, 2.0])
    @test prob isa ConvexOptimizationProblem
    @test prob isa SciMLBase.AbstractOptimizationProblem
    @test prob isa SciMLBase.AbstractSciMLProblem
    @test prob.f isa OptimizationFunction        # raw callable is wrapped
    @test prob.u0 == [1.0, 2.0]
    @test prob.p isa NullParameters
    @test prob.constraints === nothing
    @test prob.lb === nothing && prob.ub === nothing
    @test prob.sense === MinSense                # default sense
    # inplaceness follows the same convention as OptimizationProblem
    @test SciMLBase.isinplace(prob) ==
        SciMLBase.isinplace(SciMLBase.OptimizationProblem((u, p) -> sum(abs2, u), [1.0, 2.0]))

    prob2 = ConvexOptimizationProblem(
        (u, p) -> sum(u), [0.0];
        constraints = [:soc], lb = [-1.0], ub = [1.0], int = [true], sense = MaxSense
    )
    @test prob2.constraints == [:soc]
    @test prob2.sense === MaxSense
    @test prob2.lb == [-1.0] && prob2.ub == [1.0]
    @test prob2.int == [true]

    # Passing only one of lb/ub must error (parity with OptimizationProblem).
    @test_throws Exception ConvexOptimizationProblem((u, p) -> sum(u), [0.0]; lb = [-1.0])

    # Building from an explicit OptimizationFunction preserves inplaceness.
    of = OptimizationFunction((u, p) -> sum(abs2, u))
    @test ConvexOptimizationProblem(of, [1.0]) isa ConvexOptimizationProblem
end

@testset "ConvexOptimizationSolution carries duals" begin
    cache = DefaultOptimizationCache(
        OptimizationFunction((u, p) -> sum(abs2, u)), NullParameters()
    )
    sol = build_convex_solution(
        cache, :ConicBackend, [0.5, -0.5], 0.5;
        dual = [1.0, 2.0], retcode = ReturnCode.Success
    )
    @test sol isa ConvexOptimizationSolution
    @test sol isa SciMLBase.AbstractOptimizationSolution
    @test sol.u == [0.5, -0.5]
    @test sol.dual == [1.0, 2.0]
    @test sol.objective == 0.5
    @test sol.retcode == ReturnCode.Success

    # dual defaults to nothing when a backend returns no multipliers.
    sol2 = build_convex_solution(cache, :ConicBackend, [0.0], 0.0)
    @test sol2.dual === nothing
end
