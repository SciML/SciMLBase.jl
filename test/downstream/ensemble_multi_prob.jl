using ModelingToolkit, OrdinaryDiffEq, Test
using ModelingToolkit: t_nounits as t, D_nounits as D
@variables x(t), y(t)

@mtkbuild sys1 = ODESystem([D(x) ~ x,
        D(y) ~ -y], t)
@mtkbuild sys2 = ODESystem([D(x) ~ 2x,
        D(y) ~ -2y], t)
@mtkbuild sys3 = ODESystem([D(x) ~ 3x,
        D(y) ~ -3y], t)

prob1 = ODEProblem(sys1, [1.0, 1.0], (0.0, 1.0))
prob2 = ODEProblem(sys2, [2.0, 2.0], (0.0, 1.0))
prob3 = ODEProblem(sys3, [3.0, 3.0], (0.0, 1.0))

# test that when passing a vector of problems, trajectories and the prob_func are chosen appropriately
ensemble_prob = EnsembleProblem([prob1, prob2, prob3])
sol = solve(ensemble_prob, Tsit5(), EnsembleThreads())
for i in 1:3
    @test sol[1,:,i] == sol.u[i][x]
    @test sol[2,:,i] == sol.u[i][y]
end
# Ensemble is a recursive array
@test only.(sol(0.0, idxs = [x])) == sol[1, 1, :]
@test only.(sol(1.0, idxs = [x])) ≈ [sol[i][1, end] for i in 1:3]