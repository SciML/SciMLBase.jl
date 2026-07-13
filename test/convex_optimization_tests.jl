using SciMLBase, Test
using SciMLBase: MinSense, MaxSense, build_convex_solution, build_solution,
    default_calculate_dual, DefaultOptimizationCache,
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

@testset "OptimizationSolution carries duals (one unified struct)" begin
    cache = DefaultOptimizationCache(
        OptimizationFunction((u, p) -> sum(abs2, u)), NullParameters()
    )
    # Convex backend: duals on by default (Val(true)); one vector per constraint.
    sol = build_convex_solution(
        cache, :ConicBackend, [0.5, -0.5], 0.5;
        dual = [[1.0], [2.0]], retcode = ReturnCode.Success
    )
    @test sol isa OptimizationSolution            # the ONE solution struct, not a separate type
    @test sol isa SciMLBase.AbstractOptimizationSolution
    @test sol.u == [0.5, -0.5]
    @test sol.dual == [[1.0], [2.0]]
    @test sol.dual isa Vector{Vector{Float64}}    # precise dual type under Val(true)
    @test sol.objective == 0.5
    @test sol.retcode == ReturnCode.Success

    # Per-problem-type defaults: NLP off, convex on.
    @test default_calculate_dual(OptimizationProblem((u, p) -> sum(u), [0.0])) === Val(false)
    @test default_calculate_dual(ConvexOptimizationProblem((u, p) -> sum(u), [0.0])) === Val(true)

    # Val(false): duals suppressed, precise `Nothing` type (the NLP default).
    nlp = build_solution(cache, :NLP, [0.0], 0.0; calculate_dual = Val(false))
    @test nlp isa OptimizationSolution
    @test nlp.dual === nothing
    @test fieldtype(typeof(nlp), :dual) === Nothing

    # Two convex solves are the SAME concrete type (type-stable).
    solb = build_convex_solution(cache, :ConicBackend, [1.0, 1.0], 2.0; dual = [[3.0], [4.0]])
    @test typeof(sol) === typeof(solb)

    # Val(nothing) "auto": a populated dual and an absent dual share ONE concrete
    # type — the type-stable Union a DCP router relies on when it cannot know
    # statically whether the routed problem yields duals.
    autop = build_solution(cache, :Auto, [0.0], 0.0; dual = [[1.0]], calculate_dual = Val(nothing))
    auton = build_solution(cache, :Auto, [0.0], 0.0; dual = nothing, calculate_dual = Val(nothing))
    @test typeof(autop) === typeof(auton)
    @test fieldtype(typeof(autop), :dual) === Union{Nothing, Vector{Vector{Float64}}}
    @test autop.dual == [[1.0]]
    @test auton.dual === nothing
end
