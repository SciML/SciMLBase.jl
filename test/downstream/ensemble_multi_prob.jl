using ModelingToolkit, OrdinaryDiffEq, Test
using SymbolicIndexingInterface
using ModelingToolkit: t_nounits as t, D_nounits as D
@variables x(t), y(t)

@mtkcompile sys1 = System(
    [
        D(x) ~ x,
        D(y) ~ -y,
    ], t
)
@mtkcompile sys2 = System(
    [
        D(x) ~ 2x,
        D(y) ~ -2y,
    ], t
)
@mtkcompile sys3 = System(
    [
        D(x) ~ 3x,
        D(y) ~ -3y,
    ], t
)

prob1 = ODEProblem(sys1, [x => 1.0, y => 1.0], (0.0, 1.0))
prob2 = ODEProblem(sys2, [x => 2.0, y => 2.0], (0.0, 1.0))
prob3 = ODEProblem(sys3, [x => 3.0, y => 3.0], (0.0, 1.0))

# test that when passing a vector of problems, trajectories and the prob_func are chosen appropriately
ensemble_prob = EnsembleProblem([prob1, prob2, prob3])
sol = solve(ensemble_prob, Tsit5(), EnsembleThreads())
xidx = variable_index(sys1, x)
yidx = variable_index(sys1, y)
for i in 1:3
    @test sol[xidx, :, i] == sol.u[i][x]
    @test sol[yidx, :, i] == sol.u[i][y]
end
# Ensemble is a recursive array
@test only.(sol(0.0, idxs = [x])) == sol[xidx, 1, :]
@test only.(sol(1.0, idxs = [x])) ≈ [sol[i][xidx, end] for i in 1:3]
