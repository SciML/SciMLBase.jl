using ModelingToolkit, OrdinaryDiffEq, Test

@variables t, x(t), y(t)
D = Differential(t)

@named sys1 = ODESystem([D(x) ~ x,
                         D(y) ~ -y])
@named sys2 = ODESystem([D(x) ~ 2x,
                         D(y) ~ -2y])
@named sys3 = ODESystem([D(x) ~ 3x,
                         D(y) ~ -3y])

prob1 = ODEProblem(sys1, [1.0, 1.0], (0.0, 1.0))
prob2 = ODEProblem(sys2, [2.0, 2.0], (0.0, 1.0))
prob3 = ODEProblem(sys3, [3.0, 3.0], (0.0, 1.0))

# test that when passing a vector of problems, trajectories and the prob_func are chosen appropriately
ensemble_prob = EnsembleProblem([prob1, prob2, prob3])
sol = solve(ensemble_prob, Tsit5(), EnsembleThreads())
for i in 1:3
    @test sol[x, :][i] == sol[i][x]
    @test sol[y, :][i] == sol[i][y]
end
# Ensemble is a recursive array
@test Matrix(sol(0.0, idxs=[x])) == sol[1:1, 1, :] == Matrix(first(eachrow(sol[x, :]))')
# TODO: fix the interpolation
@test vec(sol(1.0, idxs=[x])) ≈ last.(sol[x, :].u)
