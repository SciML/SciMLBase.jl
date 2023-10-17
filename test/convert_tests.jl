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
        du[1] = p[1]*(u[2] - u[1])
        du[2] = u[1] * (p[2] - u[3]) - u[2]
        du[3] = u[1] * u[2] - p[3] * u[3]
    end
    u0 = [1.0; 0.0; 0.0]
    tspan = (0.0, 100.0)
    p = [10.0,28.0,8/3]
    prob = ODEProblem(lorenz!, u0, tspan,p;a=1.0,b=2.0)
    nlprob = NonlinearProblem(prob)
    @test nlprob.kwargs[:a] == prob.kwargs[:a]
    @test nlprob.kwargs[:b] == prob.kwargs[:b]
end

@testset "Convert ODEProblem with kwargs to SteadyStateProblem" begin
    function lorenz!(du, u, p, t)
        du[1] = p[1]*(u[2] - u[1])
        du[2] = u[1] * (p[2] - u[3]) - u[2]
        du[3] = u[1] * u[2] - p[3] * u[3]
    end
    u0 = [1.0; 0.0; 0.0]
    tspan = (0.0, 100.0)
    p = [10.0,28.0,8/3]
    prob = ODEProblem(lorenz!, u0, tspan,p;a=1.0,b=2.0)
    sprob = SteadyStateProblem(prob)
    @test sprob.kwargs[:a] == prob.kwargs[:a]
    @test sprob.kwargs[:b] == prob.kwargs[:b]
end