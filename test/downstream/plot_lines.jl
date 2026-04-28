# Regression tests for #1335 / OrdinaryDiffEq.jl#3573: under
# RecursiveArrayTools v4 / SciMLBase v3, `length(sol::ODESolution)` is
# `prod(size(sol)) = state_dim * n_timesteps`, not the timestep count.
# Three call sites in `solution_interface.jl` were assuming the old
# semantics, which broke both the `@recipe` Plots path and Makie's
# `convert_arguments` on any multi-state ODE.
#
# `test/solution_interface.jl` already locks the `diffeq_to_arrays`
# regression at the SciMLBase layer. This file exercises the actual
# user-visible plotting paths through the Plots recipe and the Makie
# extension, on the exact MWE from #1335 (Lorenz with `lines(sol)`).

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
# Precondition: the multi-state shape that exposes the bug.
@assert length(sol) != length(sol.t)

@testset "Plots recipe on multi-state ODESolution" begin
    using Plots: Plots, plot
    # Both the default-density path (no kwargs) and an explicit denseplot
    # path go through the @recipe definition's `plotdensity` default,
    # which used `length(sol)` instead of `length(sol.t)`.
    @test plot(sol) isa Plots.Plot
    @test plot(sol; denseplot = true) isa Plots.Plot
    @test plot(sol; denseplot = false) isa Plots.Plot
end

@testset "Makie convert_arguments on multi-state ODESolution" begin
    using Makie
    # This is the path `lines(sol)` lowers to via the Makie extension's
    # `SciMLBaseMakieExt.convert_arguments(::Type{Lines}, sol)`. No
    # backend (CairoMakie/GLMakie) needed — argument conversion runs
    # in core Makie and triggers the bug at `diffeq_to_arrays` time.
    # The extension returns a `Vector{PlotSpec}` (one Lines per series);
    # what matters here is that the call doesn't BoundsError.
    converted = Makie.convert_arguments(Makie.Lines, sol)
    @test !isempty(converted)
end
