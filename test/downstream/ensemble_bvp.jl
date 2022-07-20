using BoundaryValueDiffEq, Random

function ode!(du, u, p, t)
    du[1] = u[2]
    du[2] = -p[1] * u[1]
end

function bc!(residual, u, p, t)
    residual[1] = u[1][1] - 1.0
    residual[2] = u[end][1]
end

function prob_func(prob, i, repeat)
    remake(prob, p = [rand()])
end

initial_guess = [0.0, 1.0]
tspan = (0.0, pi / 2)
p = [rand()]
bvp = BVProblem(ode!, bc!, initial_guess, tspan, p)
ensemble_prob = EnsembleProblem(bvp, prob_func = prob_func)
sim = solve(ensemble_prob, GeneralMIRK4(), trajectories = 10, dt = 0.1)
