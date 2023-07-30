function simplependulum!(du, u, p, t)
    θ = u[1]
    dθ = u[2]
    du[1] = dθ
    du[2] = -9.81 * sin(θ)
end
function bc!(residual, u, p, t)
    residual[1] = u[1][1] + pi / 2
    residual[2] = u[end][1] - pi / 2
end
prob_bvp = BVProblem(simplependulum!, bc!, [pi / 2, pi / 2], (0, 1.0))
@test prob_bvp.tspan === (0.0, 1.0)
