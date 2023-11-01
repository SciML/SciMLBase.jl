using DifferentialEquations

f(u, p, t) = 1.01 * u
u0 = 1 / 2
tspan = (0.0, 1.0)
prob = ODEProblem(f, u0, tspan)
ensemble_prob = EnsembleProblem(prob, prob_func = (prob, i, repeat) -> remake(prob, u0 = rand()))
sim = solve(ensemble_prob, EnsembleThreads(), trajectories = 10, dt = 0.1)