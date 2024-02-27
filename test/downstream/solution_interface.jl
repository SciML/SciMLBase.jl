using ModelingToolkit, OrdinaryDiffEq, RecursiveArrayTools, StochasticDiffEq, Test
using ModelingToolkit: t_nounits as t, D_nounits as D

### Tests on non-layered model (everything should work). ###

@parameters a b c d
@variables s1(t) s2(t)

eqs = [D(s1) ~ a * s1 / (1 + s1 + s2) - b * s1,
    D(s2) ~ +c * s2 / (1 + s1 + s2) - d * s2]

@mtkbuild population_model = ODESystem(eqs,t)

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

# Tests on SDEProblem
noiseeqs = [0.1 * s1,
    0.1 * s2]
@named noisy_population_model = SDESystem(population_model, noiseeqs)
sprob = SDEProblem(complete(noisy_population_model), u0, (0.0, 100.0), p)
sol = solve(sprob, ImplicitEM())

@test sol[s1] == sol[noisy_population_model.s1] == sol[:s1]
@test sol[s2] == sol[noisy_population_model.s2] == sol[:s2]
@test_throws Exception sol[a]
@test_throws Exception sol[noisy_population_model.a]
@test_throws Exception sol[:a]
### Tests on layered model (some things should not work). ###

@parameters t σ ρ β
@variables x(t) y(t) z(t)
D = Differential(t)

eqs = [D(x) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z]

@named lorenz1 = ODESystem(eqs)
@named lorenz2 = ODESystem(eqs)

@parameters γ
@variables a(t) α(t)
connections = [0 ~ lorenz1.x + lorenz2.y + a * γ,
    α ~ 2lorenz1.x + a * γ]
@named sys = ODESystem(connections, t, [a, α], [γ], systems = [lorenz1, lorenz2])
sys_simplified = complete(structural_simplify(sys))

u0 = [lorenz1.x => 1.0,
    lorenz1.y => 0.0,
    lorenz1.z => 0.0,
    lorenz2.x => 0.0,
    lorenz2.y => 1.0,
    lorenz2.z => 0.0,
    a => 2.0]

p = [lorenz1.σ => 10.0,
    lorenz1.ρ => 28.0,
    lorenz1.β => 8 / 3,
    lorenz2.σ => 10.0,
    lorenz2.ρ => 28.0,
    lorenz2.β => 8 / 3,
    γ => 2.0]

tspan = (0.0, 100.0)
prob = ODEProblem(sys_simplified, u0, tspan, p)
sol = solve(prob, Rodas4())

@test_throws ArgumentError sol[x]
@test in(sol[lorenz1.x], [getindex.(sol.u, 1) for i in 1:length(unknowns(sol.prob.f.sys))])
@test_throws ArgumentError sol[:x]
