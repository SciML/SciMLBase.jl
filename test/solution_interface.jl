using Test, SciMLBase, RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

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

@testset "Tricks.jl inplace inference" begin
    f(u, p, t) = u
    f!(du, u, p, t) = @. du = u
    valinplace = (f) -> Val(SciMLBase.isinplace(f, 4))
    @inferred valinplace(f)
    @inferred valinplace(f!)

    @inferred ODEProblem(f, ones(3), (0.0,1.0))
    @inferred ODEProblem(f!, ones(3), (0.0,1.0))

    ex = :(function f(_du,_u,_p,_t)
        @inbounds _du[1] = _u[1]
        @inbounds _du[2] = _u[2]
        nothing
    end)
    f1 = @RuntimeGeneratedFunction(ex)
    @test_broken @inferred valinplace(f1)
end
