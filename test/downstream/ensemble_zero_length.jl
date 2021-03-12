using OrdinaryDiffEq, Test
prob = ODEProblem((u,p,t)->1.01u,0.5,(0.0,1.0), save_start=false, save_end=false)
function prob_func(prob,i,repeat)
  remake(prob,u0=rand()*prob.u0)
end
ensemble_prob = EnsembleProblem(prob,prob_func=prob_func)
sim = solve(ensemble_prob,Tsit5(),EnsembleThreads(),trajectories=10, save_everystep=false)
@test ndims(sim) == 2
@test length(sim) == 10
ts = 0.0:0.1:1.0

using SciMLBase.EnsembleAnalysis
timeseries_point_meancov(sim,ts)
