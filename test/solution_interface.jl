using Test, SciMLBase

@testset "getindex" begin
    u = rand(1)
    @test SciMLBase.build_linear_solution(nothing, u, zeros(1), nothing)[] == u[]
end

using RecipesBase: RecipesBase, apply_recipe

# No plotting backend is loaded, so accept every attribute key.
RecipesBase.is_key_supported(::Symbol) = true

function decay_solution(u0; ts = range(0.0, 1.0; length = 5))
    f = ODEFunction((u, p, t) -> -u, analytic = (u0, p, t) -> u0 .* exp(-t))
    ode = ODEProblem(f, u0, (first(ts), last(ts)))
    sol = SciMLBase.build_solution(ode, :NoAlgorithm, [first(ts)], [ode.u0])
    for t in Iterators.drop(ts, 1)
        push!(sol.t, t)
        push!(sol.u, u0 .* exp(-t))
    end
    return sol
end

function plotted_curves(sol; kwargs...)
    curves = []
    for series in apply_recipe(Dict{Symbol, Any}(kwargs), sol)
        args = series.args
        (length(args) == 2 && !isempty(args[1])) || continue
        labels = vec(series.plotattributes[:label])
        for (x, y, label) in zip(eachcol(args[1]), eachcol(args[2]), labels)
            push!(curves, (; x, y, label))
        end
    end
    return curves
end

@testset "plot analytic overlay: u0 = $u0, denseplot = $denseplot" for u0 in (
            1.0, [1.0, 2.0], [1.0 2.0; 3.0 4.0],
        ), denseplot in (false, true)
    sol = decay_solution(u0)
    names = ["u[$i]" for i in eachindex(u0)]
    curves = plotted_curves(sol; plot_analytic = true, denseplot, plotdensity = 7)
    @test [c.label for c in curves] == vcat(names, "True " .* names)
    npoints = denseplot ? 7 : length(sol.t)
    for (i, name) in enumerate(names)
        numerical = only(filter(c -> c.label == name, curves))
        analytic = only(filter(c -> c.label == "True " * name, curves))
        @test length(analytic.x) == npoints
        @test analytic.x == numerical.x
        @test analytic.y ≈ u0[i] .* exp.(-analytic.x)
        @test numerical.y ≈ [sol(t)[i] for t in numerical.x]
    end
end

@testset "plot animation frame: u0 = $u0, denseplot = $denseplot, plot_analytic = $plot_analytic" for u0 in (
            1.0, [1.0, 2.0],
        ), denseplot in (false, true), plot_analytic in (false, true)
    sol = decay_solution(u0)
    for tslocation in (1, 3)
        frame = SciMLBase.solution_new_tslocation(sol, tslocation)
        curves = plotted_curves(frame; denseplot, plot_analytic)
        @test length(curves) == length(u0) * (plot_analytic ? 2 : 1)
        for c in curves
            @test first(c.x) == sol.t[1]
            @test last(c.x) == sol.t[tslocation]
        end
    end
end

@testset "plot without continuous variables: u0 = $u0, plot_analytic = $plot_analytic" for u0 in (
            1.0, [1.0, 2.0], [1.0 2.0; 3.0 4.0],
        ), plot_analytic in (false, true)
    sol = decay_solution(u0)
    @test isempty(plotted_curves(sol; idxs = Tuple[], plot_analytic))
end

@testset "sparse plotting time limits: step = $(step(ts))" for ts in (0.0:0.25:1.0, 1.0:-0.25:0.0)
    sol = decay_solution([1.0, 2.0]; ts)
    frame = SciMLBase.solution_new_tslocation(sol, 3)
    @testset "tspan = $tspan" for tspan in ((0.25, 0.75), (0.75, 0.25))
        for (plotted_sol, saved_times) in ((sol, ts), (frame, ts[1:3]))
            expected = filter(t -> 0.25 <= t <= 0.75, saved_times)
            curves = plotted_curves(plotted_sol; denseplot = false, plot_analytic = true, tspan)
            @test length(curves) == 4
            @test all(c -> c.x == expected, curves)
        end
    end
    @test_throws ArgumentError plotted_curves(sol; denseplot = false, tspan = (0.1, 0.2))
end

@testset "plot axis guides" begin
    sol = decay_solution([1.0, 2.0])
    series = only(apply_recipe(Dict{Symbol, Any}(:idxs => (1, 2)), sol))
    @test series.plotattributes[:xguide] == "u[1]"
    @test series.plotattributes[:yguide] == "u[2]"
    series = only(apply_recipe(Dict{Symbol, Any}(:idxs => (0, 2)), sol))
    @test series.plotattributes[:xguide] == "t"
    @test series.plotattributes[:yguide] == "u[2]"
    for (idxs, guide) in (([(1, 0), (2, 0)], :yguide), ([(1, 2, 0)], :zguide))
        series = only(apply_recipe(Dict{Symbol, Any}(:idxs => idxs), sol))
        @test get(series.plotattributes, guide, nothing) == "t"
    end
end

@testset "sparse analytic overlay uses saved values" begin
    ts = collect(range(0.0, 1.0; length = 5))
    p = Ref(1.0)
    f = ODEFunction((u, p, t) -> -u, analytic = (u0, p, t) -> p[] * u0 * exp(-t))
    prob = ODEProblem(f, 1.0, (first(ts), last(ts)), p)
    sol = SciMLBase.build_solution(prob, :NoAlgorithm, ts, exp.(-ts))
    p[] = 2.0

    for plotted_sol in (sol, SciMLBase.solution_new_tslocation(sol, 3))
        numerical, analytic = plotted_curves(plotted_sol; denseplot = false, plot_analytic = true)
        @test analytic.y ≈ exp.(-numerical.x)
    end

    _, analytic = plotted_curves(sol; plotat = [0.1, 0.9], plot_analytic = true)
    @test analytic.y ≈ 2 .* exp.(-analytic.x)
end

@testset "plot DDE analytic overlay" begin
    ts = collect(range(0.0, 1.0; length = 5))
    h = (p, t) -> 1.0
    f = DDEFunction((u, h, p, t) -> -u; analytic = (u0, h, p, t) -> h(p, 0) * exp(-t))
    prob = DDEProblem(f, 1.0, h, (first(ts), last(ts)))
    sol = SciMLBase.build_solution(prob, :NoAlgorithm, ts, exp.(-ts))
    for denseplot in (false, true)
        _, analytic = plotted_curves(sol; plot_analytic = true, denseplot, plotdensity = 7)
        @test analytic.y ≈ exp.(-analytic.x)
    end
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
        ODEProblem(f, 1.0, (0.0, 1.0)), :NoAlgorithm, 0.0:0.1:1.0, exp.(0.0:0.1:1.0)
    )
    sol2 = SciMLBase.build_solution(
        ODEProblem(f, [1.0, 2.0], (0.0, 1.0)), :NoAlgorithm,
        0.0:0.1:1.0, vcat.(exp.(0.0:0.1:1.0), 2exp.(0.0:0.1:1.0))
    )
    for sol in [sol1, sol2]
        @test sol(0.15; idxs = []) == Float64[]
        @test sol(0.15; idxs = Int[]) == Float64[]
        @test sol([0.15, 0.25]; idxs = []) == [Float64[], Float64[]]
        @test sol([0.15, 0.25]; idxs = Int[]) == [Float64[], Float64[]]
    end
end
