using OrdinaryDiffEq, Test, LinearAlgebra
prob = ODEProblem((u, p, t) -> 1.01u, 0.5, (0.0, 1.0), save_start = false, save_end = false)
function prob_func(prob, i, repeat)
    remake(prob, u0 = rand() * prob.u0)
end
ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
sim = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories = 10,
    save_everystep = false)
@test ndims(sim) == 2
@test length(sim.u) == 10
@test eltype(sim.u) <: ODESolution
ts = 0.0:0.1:1.0

using SciMLBase.EnsembleAnalysis
sim = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories = 10, saveat = 0.1)
timeseries_point_meancov(sim, ts)
timeseries_point_median(sim, ts)

function prob_sol(_p)
    prob = ODEProblem((u, p, t) -> p .* u, _p, (0.0, 1.0), _p, save_start = false,
        save_end = false)
    sim = solve(prob, Tsit5())
end
mapres = SciMLBase.responsible_map(prob_sol, [0.5, diagm([1.0, 1.0])])
@test length(mapres) == 2
