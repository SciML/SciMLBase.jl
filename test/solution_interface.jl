using Test, SciMLBase

@testset "getindex" begin
    u = rand(1)
    @test SciMLBase.build_linear_solution(nothing, u, zeros(1), nothing)[] == u[]
end

@testset "plot ODE solution" begin
    f = ODEFunction((u, p, t) -> -u, analytic = (u0, p, t) -> u0 * exp(-t))
    ode = ODEProblem(f, 1.0, (0.0, 1.0))
    sol = SciMLBase.build_solution(ode, :NoAlgorithm, [ode.tspan[begin]], [ode.u0])
    for t in Iterators.drop(range(ode.tspan..., length = 5), 1)
        push!(sol.t, t)
        push!(sol.u, ode.u0)
    end

    int_vars = SciMLBase.interpret_vars(nothing, sol) # nothing = idxs
    plot_vecs,
    labels = SciMLBase.diffeq_to_arrays(sol,
        true, # plot_analytic
        true, # denseplot
        10, # plotdensity
        ode.tspan,
        int_vars,
        :identity,
        nothing) # tscale
    @test plot_vecs[2][:, 2] ≈ @. exp(-plot_vecs[1][:, 2])
end

@testset "interpolate empty ODE solution" begin
    f = (u, p, t) -> -u
    ode = ODEProblem(f, 1.0, (0.0, 1.0))
    sol = SciMLBase.build_solution(ode, :NoAlgorithm, [ode.tspan[begin]], [ode.u0])
    @test sol(0.0) == 1.0
    @test sol([0.0, 0.0]) == [1.0, 1.0]
    # test that indexing out of bounds doesn't segfault
    @test_throws ErrorException sol(1)
    @test_throws ErrorException sol(-0.5)
    @test_throws ErrorException sol([0, -0.5, 0])
end

@testset "interpolate with empty idxs" begin
    f = (u, p, t) -> u
    sol1 = SciMLBase.build_solution(
        ODEProblem(f, 1.0, (0.0, 1.0)), :NoAlgorithm, 0.0:0.1:1.0, exp.(0.0:0.1:1.0))
    sol2 = SciMLBase.build_solution(ODEProblem(f, [1.0, 2.0], (0.0, 1.0)), :NoAlgorithm,
        0.0:0.1:1.0, vcat.(exp.(0.0:0.1:1.0), 2exp.(0.0:0.1:1.0)))
    for sol in [sol1, sol2]
        @test sol(0.15; idxs = []) == Float64[]
        @test sol(0.15; idxs = Int[]) == Float64[]
        @test sol([0.15, 0.25]; idxs = []) == [Float64[], Float64[]]
        @test sol([0.15, 0.25]; idxs = Int[]) == [Float64[], Float64[]]
    end
end
