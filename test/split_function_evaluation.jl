using Test, SciMLBase

# https://github.com/SciML/DifferentialEquations.jl/issues/1041
function f1!(du, u, p, t)
    du .= 1.0
    return nothing
end

# leaves the second component to whatever was already in `du`
function f2!(du, u, p, t)
    du[1] = 10.0
    return nothing
end

function g!(du, u, p, t)
    du .= 0.0
    return nothing
end

@testset "SplitFunction with an f2 that writes part of du" begin
    prob = SplitODEProblem(f1!, f2!, zeros(2), (0.0, 1.0))
    du = [-5.0, -5.0]
    prob.f(du, prob.u0, prob.p, 0.0)
    @test du == [11.0, 1.0]
    prob.f(du, prob.u0, prob.p, 0.0)
    @test du == [11.0, 1.0]
end

@testset "SplitSDEFunction with an f2 that writes part of du" begin
    prob = SplitSDEProblem(f1!, f2!, g!, zeros(2), (0.0, 1.0))
    du = [-5.0, -5.0]
    prob.f(du, prob.u0, prob.p, 0.0)
    @test du == [11.0, 1.0]
    prob.f(du, prob.u0, prob.p, 0.0)
    @test du == [11.0, 1.0]
end
