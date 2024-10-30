using OrdinaryDiffEq, Test

A = [1 2
     3 4]
prob = ODEProblem((u, p, t) -> A*u, ones(2,2), (0.0, 1.0))
function prob_func(prob, i, repeat)
    remake(prob, u0 = i * prob.u0)
end
ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
sim = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories = 10, saveat=0.01)
@test sim isa EnsembleSolution
@test size(sim[1,:,:,:])  == (2,101,10)
@test size(sim[:,1,:,:]) == (2,101,10)
@test size(sim[:,:,1,:]) == (2,2,10)
@test size(sim[:,:,:,1]) == (2,2,101)
@test Array(sim)[1,:,:,:]  == sim[1,:,:,:]
@test Array(sim)[:,1,:,:]  == sim[:,1,:,:]
@test Array(sim)[:,:,1,:]  == sim[:,:,1,:]
@test Array(sim)[:,:,:,1]  == sim[:,:,:,1]