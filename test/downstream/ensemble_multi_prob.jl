using OrdinaryDiffEq, Test

prob1 = ODEProblem((u, p, t) -> 0.99u, 0.55, (0.0, 1.1))
prob1 = ODEProblem((u, p, t) -> 1.0u, 0.45, (0.0, 0.9))
output_func(sol, i) = (last(sol), false)
# test that when passing a vector of problems, trajectories and the prob_func are chosen appropriately
ensemble_prob = EnsembleProblem([prob1, prob2], output_func = output_func)
sim = solve(ensemble_prob, Tsit5(), EnsembleThreads())
