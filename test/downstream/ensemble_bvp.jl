using BoundaryValueDiffEq, Random

function ode!(du, u, p, t)
    du[1] = u[2]
    return du[2] = -p[1] * u[1]
end

function bc!(residual, u, p, t)
    residual[1] = u[1][1] - 1.0
    return residual[2] = u[end][1]
end

function prob_func(prob, i, repeat)
    return remake(prob, p = [rand()])
end

initial_guess = [0.0, 1.0]
tspan = (0.0, pi / 2)
p = [rand()]
bvp = BVProblem(ode!, bc!, initial_guess, tspan, p)
ensemble_prob = EnsembleProblem(bvp, prob_func = prob_func)
sim = solve(ensemble_prob, MIRK4(), trajectories = 10, dt = 0.1)
