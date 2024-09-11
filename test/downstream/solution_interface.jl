using ModelingToolkit, OrdinaryDiffEq, RecursiveArrayTools, StochasticDiffEq, Test
using ModelingToolkit: t_nounits as t, D_nounits as D
using Plots: Plots, plot

### Tests on non-layered model (everything should work). ###

@parameters a b c d
@variables s1(t) s2(t)

eqs = [D(s1) ~ a * s1 / (1 + s1 + s2) - b * s1,
    D(s2) ~ +c * s2 / (1 + s1 + s2) - d * s2]

@mtkbuild population_model = ODESystem(eqs, t)

# Tests on ODEProblem.
u0 = [s1 => 2.0, s2 => 1.0]
p = [a => 2.0, b => 1.0, c => 1.0, d => 1.0]
tspan = (0.0, 1000000.0)
oprob = ODEProblem(population_model, u0, tspan, p)
sol = solve(oprob, Rodas4())

@test sol[s1] == sol[population_model.s1] == sol[:s1]
@test sol[s2] == sol[population_model.s2] == sol[:s2]
@test sol[s1][end] ≈ 1.0
@test_throws Exception sol[a]
@test_throws Exception sol[population_model.a]
@test_throws Exception sol[:a]

@testset "plot ODE solution" begin
    Plots.unicodeplots()
    f = ODEFunction((u, p, t) -> -u, analytic = (u0, p, t) -> u0 * exp(-t))

    # scalar
    ode = ODEProblem(f, 1.0, (0.0, 1.0))
    sol = solve(ode, Tsit5())
    @test_nowarn plot(sol)
    @test_nowarn plot(sol; plot_analytic = true)

    # vector
    ode = ODEProblem(f, [1.0, 2.0], (0.0, 1.0))
    sol = solve(ode, Tsit5())
    @test_nowarn plot(sol)
    @test_nowarn plot(sol; plot_analytic = true)

    # matrix
    ode = ODEProblem(f, [1.0 2.0; 3.0 4.0], (0.0, 1.0))
    sol = solve(ode, Tsit5())
    @test_nowarn plot(sol)
    @test_nowarn plot(sol; plot_analytic = true)
end

# Tests on SDEProblem
noiseeqs = [0.1 * s1,
    0.1 * s2]
@named noisy_population_model = SDESystem(population_model, noiseeqs)
noisy_population_model = complete(noisy_population_model)
sprob = SDEProblem(noisy_population_model, u0, (0.0, 100.0), p)
sol = solve(sprob, ImplicitEM())

@test sol[s1] == sol[noisy_population_model.s1] == sol[:s1]
@test sol[s2] == sol[noisy_population_model.s2] == sol[:s2]
@test_throws Exception sol[a]
@test_throws Exception sol[noisy_population_model.a]
@test_throws Exception sol[:a]
@test_nowarn sol(0.5, idxs = noisy_population_model.s1)
### Tests on layered model (some things should not work). ###

@parameters σ ρ β
@variables x(t) y(t) z(t)

eqs = [D(x) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z]

@named lorenz1 = ODESystem(eqs, t)
@named lorenz2 = ODESystem(eqs, t)

@parameters γ
@variables a(t) α(t)
connections = [0 ~ lorenz1.x + lorenz2.y + a * γ,
    α ~ 2lorenz1.x + a * γ]
@mtkbuild sys = ODESystem(connections, t, [a, α], [γ], systems = [lorenz1, lorenz2])

u0 = [lorenz1.x => 1.0,
    lorenz1.y => 0.0,
    lorenz1.z => 0.0,
    lorenz2.x => 0.0,
    lorenz2.y => 1.0,
    lorenz2.z => 0.0]

p = [lorenz1.σ => 10.0,
    lorenz1.ρ => 28.0,
    lorenz1.β => 8 / 3,
    lorenz2.σ => 10.0,
    lorenz2.ρ => 28.0,
    lorenz2.β => 8 / 3,
    γ => 2.0]

tspan = (0.0, 100.0)
prob = ODEProblem(sys, u0, tspan, p)
sol = solve(prob, Rodas4())

@test_throws ArgumentError sol[x]
@test in(sol[lorenz1.x], [getindex.(sol.u, 1) for i in 1:length(unknowns(sol.prob.f.sys))])
@test_throws KeyError sol[:x]

### Non-symbolic indexing tests
@test sol[:, 1] isa AbstractVector
@test sol[:, 1:2] isa AbstractDiffEqArray
@test sol[:, [1, 2]] isa AbstractDiffEqArray

sol1 = sol(0.0:1.0:10.0)
@test sol1.u isa Vector
@test first(sol1.u) isa Vector
@test length(sol1.u) == 11
@test length(sol1.t) == 11

sol2 = sol(0.1)
@test sol2 isa Vector
@test length(sol2) == length(unknowns(sys))
@test first(sol2) isa Real

sol3 = sol(0.0:1.0:10.0, idxs = [lorenz1.x, lorenz2.x])

sol7 = sol(0.0:1.0:10.0, idxs = [2, 1])
@test sol7.u isa Vector
@test first(sol7.u) isa Vector
@test length(sol7.u) == 11
@test length(sol7.t) == 11
@test collect(sol7[t]) ≈ sol3.t
@test collect(sol7[t, 1:5]) ≈ sol3.t[1:5]

sol8 = sol(0.1, idxs = [2, 1])
@test sol8 isa Vector
@test length(sol8) == 2
@test first(sol8) isa Real

sol9 = sol(0.0:1.0:10.0, idxs = 2)
@test sol9.u isa Vector
@test first(sol9.u) isa Real
@test length(sol9.u) == 11
@test length(sol9.t) == 11
@test collect(sol9[t]) ≈ sol3.t
@test collect(sol9[t, 1:5]) ≈ sol3.t[1:5]

sol10 = sol(0.1, idxs = 2)
@test sol10 isa Real

@testset "Plot idxs" begin
    @variables x(t) y(t)
    @parameters p
    @mtkbuild sys = ODESystem([D(x) ~ x * t, D(y) ~ y - p * x], t)
    prob = ODEProblem(sys, [x => 1.0, y => 2.0], (0.0, 1.0), [p => 1.0])
    sol = solve(prob, Tsit5())

    plotfn(t, u) = (t, 2u)
    all_idxs = [x, x + p * y, t, (plotfn, 0, 1), (plotfn, t, 1), (plotfn, 0, x),
        (plotfn, t, x), (plotfn, t, p * y)]
    sym_idxs = [:x, :t, (plotfn, :t, 1), (plotfn, 0, :x),
        (plotfn, :t, :x)]
    for idx in Iterators.flatten((all_idxs, sym_idxs))
        @test_nowarn plot(sol; idxs = idx)
        @test_nowarn plot(sol; idxs = [idx])
    end
    for idx in Iterators.flatten((
        Iterators.product(all_idxs, all_idxs), Iterators.product(sym_idxs, sym_idxs)))
        @test_nowarn plot(sol; idxs = collect(idx))
        if !(idx[1] isa Tuple || idx[2] isa Tuple)
            @test_nowarn plot(sol; idxs = idx)
        end
    end
end
