using OrdinaryDiffEq, Test, Statistics

# test for https://github.com/SciML/SciMLBase.jl/issues/190
prob = ODEProblem((u, p, t) -> 1.01u, 0.5, (0.0, 1.0))
prob_func(prob, i, repeat) = remake(prob, u0 = rand() * prob.u0)
output_func(sol, i) = (last(sol), false)
reduction(u, batch, I) = (append!(u, mean(batch)), false)
# make sure first batch is timed (test using 1 batch but reduction)
ensemble_prob = EnsembleProblem(prob, prob_func = prob_func, output_func = output_func,
    reduction = reduction)
sim = solve(ensemble_prob, Tsit5(), EnsembleThreads(), batch_size = 2)
