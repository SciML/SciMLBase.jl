using OrdinaryDiffEq

prob = ODEProblem((u, p, t) -> 1.01u, 0.5, (0.0, 1.0))
function prob_func(prob, i, repeat)
    remake(prob, u0 = rand() * prob.u0)
end
ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
sim = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories = 10)
@test sim isa EnsembleSolution
