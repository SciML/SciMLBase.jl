using Test, SciMLBase

@testset "Convert NonlinearFunction to ODEFunction" begin
    f! = NonlinearFunction((du, u, p) -> du[1] = u[1] - p[1] + p[2])
    f = NonlinearFunction((u, p) -> u .- p[1] .+ p[2])

    _f! = ODEFunction(f!)
    _f = ODEFunction(f)

    @test _f! isa ODEFunction && isinplace(_f!)
    @test _f isa ODEFunction && !isinplace(_f)
end

@testset "Convert ODEFunction to NonlinearFunction" begin
    f! = ODEFunction((du, u, p, t) -> du[1] = u[1] - p[1] + p[2])
    f = ODEFunction((u, p, t) -> u .- p[1] .+ p[2])

    _f! = NonlinearFunction(f!)
    _f = NonlinearFunction(f)

    @test _f! isa NonlinearFunction && isinplace(_f!)
    @test _f isa NonlinearFunction && !isinplace(_f)
end

@testset "Convert ODEProblem to NonlinearProblem" begin
    function lorenz!(du, u, p, t)
        du[1] = 10.0(u[2] - u[1])
        du[2] = u[1] * (28.0 - u[3]) - u[2]
        du[3] = u[1] * u[2] - (8 / 3) * u[3]
    end
    u0 = [1.0; 0.0; 0.0]
    tspan = (0.0, 100.0)
    prob = ODEProblem(lorenz!, u0, tspan)
    nlprob = NonlinearProblem(prob)
end

@testset "Convert ODEProblem with kwargs to NonlinearProblem" begin
    function lorenz!(du, u, p, t)
        du[1] = p[1] * (u[2] - u[1])
        du[2] = u[1] * (p[2] - u[3]) - u[2]
        du[3] = u[1] * u[2] - p[3] * u[3]
    end
    u0 = [1.0; 0.0; 0.0]
    tspan = (0.0, 100.0)
    p = [10.0, 28.0, 8 / 3]
    prob = ODEProblem(lorenz!, u0, tspan, p; a = 1.0, b = 2.0)
    nlprob = NonlinearProblem(prob)
    @test nlprob.kwargs[:a] == prob.kwargs[:a]
    @test nlprob.kwargs[:b] == prob.kwargs[:b]
end

@testset "unwrapped_f preserves non-nothing fields for AutoSpecialize" begin
    analytic_fn = (u0, p, t) -> u0 .* exp(-t)
    jvp_fn = (Jv, u, p, t, v) -> nothing

    # AutoSpecialize with analytic + jvp should not error
    f = ODEFunction{true, SciMLBase.AutoSpecialize}(
        (du, u, p, t) -> (du .= -u);
        analytic = analytic_fn,
        jvp = jvp_fn
    )
    uf = SciMLBase.unwrapped_f(f)
    @test uf.analytic === analytic_fn
    @test uf.jvp === jvp_fn

    # AutoSpecialize with only analytic
    f2 = ODEFunction{true, SciMLBase.AutoSpecialize}(
        (du, u, p, t) -> (du .= -u);
        analytic = analytic_fn
    )
    uf2 = SciMLBase.unwrapped_f(f2)
    @test uf2.analytic === analytic_fn
    @test uf2.jvp === nothing

    # AutoSpecialize with no extra fields
    f3 = ODEFunction{true, SciMLBase.AutoSpecialize}(
        (du, u, p, t) -> (du .= -u)
    )
    uf3 = SciMLBase.unwrapped_f(f3)
    @test uf3.analytic === nothing
    @test uf3.jvp === nothing
end

@testset "Convert ODEProblem with kwargs to SteadyStateProblem" begin
    function lorenz!(du, u, p, t)
        du[1] = p[1] * (u[2] - u[1])
        du[2] = u[1] * (p[2] - u[3]) - u[2]
        du[3] = u[1] * u[2] - p[3] * u[3]
    end
    u0 = [1.0; 0.0; 0.0]
    tspan = (0.0, 100.0)
    p = [10.0, 28.0, 8 / 3]
    prob = ODEProblem(lorenz!, u0, tspan, p; a = 1.0, b = 2.0)
    sprob = SteadyStateProblem(prob)
    @test sprob.kwargs[:a] == prob.kwargs[:a]
    @test sprob.kwargs[:b] == prob.kwargs[:b]
end
