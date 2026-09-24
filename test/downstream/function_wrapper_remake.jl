using OrdinaryDiffEq, SciMLBase, Test

# An out-of-place `FunctionWrapperSpecialize` function is wrapped for `(u, p, t)`, so
# `remake` has to rewrap it with that signature rather than the in-place `(du, u, p, t)`.
f(u, p, t) = -p[1] .* u
prob = ODEProblem{false, SciMLBase.FunctionWrapperSpecialize}(f, [1.0], (0.0, 1.0), [2.0])
remade = remake(prob; p = [3.0])
@test SciMLBase.specialization(remade.f) === SciMLBase.FunctionWrapperSpecialize
@test remade.f(remade.u0, remade.p, 0.0) == [-3.0]

sol = solve(remade, Tsit5())
@test SciMLBase.successful_retcode(sol)
fresh = ODEProblem{false, SciMLBase.FunctionWrapperSpecialize}(f, [1.0], (0.0, 1.0), [3.0])
@test sol.u[end] ≈ solve(fresh, Tsit5()).u[end]
