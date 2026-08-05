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

@testset "tspan crops the plotted series" begin
    using Plots: Plots, plot
    sparse_sol = solve(prob, Tsit5(), dense = false)
    window = (10.0, 20.0)

    for (s, dense) in ((sol, true), (sol, false), (sparse_sol, false))
        x = plot(s; idxs = 1, denseplot = dense, tspan = window).series_list[1][:x]
        @test !isempty(x)
        @test all(t -> window[1] <= t <= window[2], x)
    end

    # A sparse plot draws exactly the saved points inside the window
    x = plot(sparse_sol; idxs = 1, tspan = window).series_list[1][:x]
    @test x == filter(t -> window[1] <= t <= window[2], sparse_sol.t)

    @test_throws ArgumentError plot(sparse_sol; idxs = 1, tspan = (200.0, 300.0))
end

@testset "tspan crops solutions integrated backwards in time" begin
    using Plots: Plots, plot
    decay_prob = ODEProblem((u, p, t) -> -0.01 * u, 1.0, (100.0, 0.0))
    backwards_sol = solve(decay_prob, Tsit5(), dense = false, saveat = 1.0)

    for window in ((80.0, 20.0), (20.0, 80.0))
        x = plot(backwards_sol; tspan = window).series_list[1][:x]
        @test extrema(x) == (20.0, 80.0)
    end
end
