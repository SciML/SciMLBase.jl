using OrdinaryDiffEq
using Test

f(u,p,t) = 1.01*u
u0=1/2
tspan = (0.0,1.0)
prob = ODEProblem(f,u0,tspan)
function prob_func(prob, i, repeat)
    remake(prob, u0 = rand() * prob.u0)
end
ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
sim = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories = 10)
@test sim.stats.nf == mapreduce(x -> x.stats.nf, +, sim.u)