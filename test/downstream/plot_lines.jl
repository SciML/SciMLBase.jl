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

using ModelingToolkit
using ModelingToolkit: t_nounits as tiv, D_nounits as Dt
using Plots: Plots, plot
using SciMLBase: symbolic_interpolation

@parameters a b c
@variables xs(tiv) ys(tiv) zs(tiv) ws(tiv)
@mtkcompile mtksys = System(
    [
        Dt(xs) ~ a * (ys - xs)
        Dt(ys) ~ xs * (b - zs) - ys
        Dt(zs) ~ xs * ys - c * zs
        ws ~ xs + ys + zs
    ], tiv
)
mtkprob = ODEProblem(
    mtksys, [xs => 1.0, ys => 0.0, zs => 0.0, a => 10.0, b => 28.0, c => 8 / 3],
    (0.0, 10.0)
)
mtksol = solve(mtkprob, Tsit5())

series_labels(p) = [s[:label] for s in p.series_list]

@testset "Integrator recipe without symbolic metadata" begin
    integ = init(prob, Tsit5())
    step!(integ, 1.0, true)

    for idxs in (nothing, 1, [1, 2], (1, 2))
        kwargs = idxs === nothing ? (;) : (; idxs)
        @test plot(integ; kwargs...) isa Plots.Plot
        @test plot(integ; denseplot = false, kwargs...) isa Plots.Plot
    end
    @test series_labels(plot(integ)) == series_labels(plot(sol))
end

@testset "Integrator recipe resolves observed equations" begin
    integ = init(mtkprob, Tsit5())
    step!(integ, 1.0, true)
    plotted_t = collect(range(integ.tprev, integ.t; length = 10))

    # `ws` is an observed equation, so it is not a component of `integrator.u`
    @test SciMLBase.is_observed(integ, ws)

    for idxs in (nothing, xs, ws, [xs, ws], (xs, ws))
        kwargs = idxs === nothing ? (;) : (; idxs)
        @test plot(integ; kwargs...) isa Plots.Plot
        @test plot(integ; denseplot = false, kwargs...) isa Plots.Plot
        # Series labels match what the same specification gives for a solution
        @test series_labels(plot(integ; kwargs...)) ==
            series_labels(plot(mtksol; kwargs...))
    end

    # The plotted values are the interpolated observed values, not raw state
    @test plot(integ; idxs = ws).series_list[1][:y] ≈
        symbolic_interpolation(integ, plotted_t, ws).u
    @test plot(integ; idxs = ws).series_list[1][:y] ≈ [sum(integ(tt)) for tt in plotted_t]

    # Scatter of the current point only
    sparse_series = plot(integ; denseplot = false, idxs = ws).series_list[1]
    @test sparse_series[:x] == [integ.t]
    @test sparse_series[:y] ≈ [integ[ws]]

    # A user-supplied plot function is applied, as it is for solutions
    scaled = plot(integ; idxs = ((tt, wv) -> (tt, 2wv), 0, ws)).series_list[1]
    @test scaled[:y] ≈ 2 .* plot(integ; idxs = ws).series_list[1][:y]

    @test_throws ArgumentError plot(integ; plot_analytic = true)
end

@testset "symbolic_interpolation on an integrator" begin
    integ = init(mtkprob, Tsit5())
    step!(integ, 1.0, true)
    mid = (integ.tprev + integ.t) / 2

    @test symbolic_interpolation(integ, mid, ws) ≈
        sum(symbolic_interpolation(integ, mid, [xs, ys, zs]))
    @test symbolic_interpolation(integ, integ.t, ws) ≈ integ[ws]
    batched = symbolic_interpolation(integ, [mid, integ.t], [xs, ws])
    @test batched.t == [mid, integ.t]
    @test batched.u == [symbolic_interpolation(integ, tt, [xs, ws]) for tt in (mid, integ.t)]
    # Rows are the requested quantities in order, as they are for `sol(t; idxs)`
    @test batched[1, :] ≈ [symbolic_interpolation(integ, tt, xs) for tt in (mid, integ.t)]
    @test batched[2, :] ≈ [symbolic_interpolation(integ, tt, ws) for tt in (mid, integ.t)]
    @test_throws ErrorException symbolic_interpolation(integ, mid, ws, Val{1})
end
