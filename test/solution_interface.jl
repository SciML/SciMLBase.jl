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

    syms = SciMLBase.interpret_vars(nothing, sol, SciMLBase.getsyms(sol))
    int_vars = SciMLBase.interpret_vars(nothing, sol, syms) # nothing = idxs
    plot_vecs, labels = SciMLBase.diffeq_to_arrays(sol,
        true, # plot_analytic
        true, # denseplot
        10, # plotdensity
        ode.tspan,
        0.1, # axis_safety
        nothing, # idxs
        int_vars,
        :identity, # tscale
        nothing) # strs
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
