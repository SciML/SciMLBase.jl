using OrdinaryDiffEq, Test

function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
    return nothing
end
prob = ODEProblem(lorenz!, [1.0, 0.0, 0.0], (0.0, 100.0), (10.0, 28.0, 8 / 3))
sol = solve(prob, Tsit5())

@testset "Plots recipe on multi-state ODESolution" begin
    using Plots: Plots, plot
    @test plot(sol) isa Plots.Plot
    @test plot(sol; denseplot = true) isa Plots.Plot
    @test plot(sol; denseplot = false) isa Plots.Plot
end

@testset "Makie convert_arguments on multi-state ODESolution" begin
    using Makie
    converted = Makie.convert_arguments(Makie.Lines, sol)
    @test !isempty(converted)
end
