using OrdinaryDiffEq, SciMLBase

function lorenz!(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end

u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 1.0)
prob = ODEProblem(lorenz!, u0, tspan)
# implicit solver so we can test cache stripping worked
sol = solve(prob, Rosenbrock23())

stripped_sol = SciMLBase.strip_solution(sol)

@test isnothing(SciMLBase.strip_solution(sol, strip_alg = true).alg)

@test isnothing(stripped_sol.interp.f)

@test isnothing(stripped_sol.interp.cache.jac_config)

@test isnothing(stripped_sol.interp.cache.grad_config)
