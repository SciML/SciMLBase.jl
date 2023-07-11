using ModelingToolkit, OrdinaryDiffEq, Test

@variables t, x(t)
D = Differential(t)

@named sys1 = ODESystem([D(x) ~ 1.1*x])
@named sys2 = ODESystem([D(x) ~ 1.2*x])

prob1 = ODEProblem(sys1, [2.0], (0.0, 1.0))
prob2 = ODEProblem(sys2, [1.0], (0.0, 1.0))

# test that when passing a vector of problems, trajectories and the prob_func are chosen appropriately
ensemble_prob = EnsembleProblem([prob1, prob2])
sol = solve(ensemble_prob, Tsit5(), EnsembleThreads())
@test isapprox(sol[x], [2,1] .* map(Base.Fix1(map, exp), [1.1, 1.2] .* sol[t]), rtol=1e-4)
# Ensemble is a recursive array
@test sol(0.0, idxs=[x]) == sol[:, 1] == first.(sol[x], 1)
@test sol(1.0, idxs=[x]) == sol[:, end] == last.(sol[x], 1)
