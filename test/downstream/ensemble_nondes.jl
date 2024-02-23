using Optimization, OptimizationOptimJL, ForwardDiff, Test

x0 = zeros(2)
rosenbrock(x, p = nothing) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
l1 = rosenbrock(x0)

optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0)
sol1 = Optimization.solve(prob, OptimizationOptimJL.BFGS(), maxiters = 5)

ensembleprob = Optimization.EnsembleProblem(
    prob, [x0, x0 .+ rand(2), x0 .+ rand(2), x0 .+ rand(2)])

sol = Optimization.solve(ensembleprob, OptimizationOptimJL.BFGS(),
    EnsembleThreads(), trajectories = 4, maxiters = 5)
@test findmin(i -> sol[i].objective, 1:4)[1] < sol1.objective

sol = Optimization.solve(ensembleprob, OptimizationOptimJL.BFGS(),
    EnsembleDistributed(), trajectories = 4, maxiters = 5)
@test findmin(i -> sol[i].objective, 1:4)[1] < sol1.objective

prob = OptimizationProblem(optf, x0, lb = [-0.5, -0.5], ub = [0.5, 0.5])
ensembleprob = Optimization.EnsembleProblem(
    prob, prob_func = (prob, i, repeat) -> remake(prob, u0 = rand(-0.5:0.001:0.5, 2)))

sol = Optimization.solve(ensembleprob, OptimizationOptimJL.BFGS(),
    EnsembleThreads(), trajectories = 5, maxiters = 5)
@test findmin(i -> sol[i].objective, 1:4)[1] < sol1.objective

sol = Optimization.solve(ensembleprob, OptimizationOptimJL.BFGS(),
    EnsembleDistributed(), trajectories = 5, maxiters = 5)
@test findmin(i -> sol[i].objective, 1:4)[1] < sol1.objective

using NonlinearSolve

f(u, p) = u .* u .- p
u0 = [1.0, 1.0]
p = 2.0
prob = NonlinearProblem(f, u0, p)
ensembleprob = EnsembleProblem(prob, [u0, u0 .+ rand(2), u0 .+ rand(2), u0 .+ rand(2)])

sol = solve(ensembleprob, EnsembleThreads(), trajectories = 4, maxiters = 100)

sol = solve(ensembleprob, EnsembleDistributed(), trajectories = 4, maxiters = 100)
