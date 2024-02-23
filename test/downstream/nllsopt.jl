using NonlinearSolve, Optimization, OptimizationNLopt, ForwardDiff

true_function(x, θ) = @. θ[1] * exp(θ[2] * x) * cos(θ[3] * x + θ[4])

θ_true = [1.0, 0.1, 2.0, 0.5]

x = [-1.0, -0.5, 0.0, 0.5, 1.0]

y_target = true_function(x, θ_true)

function loss_function(θ, p)
    ŷ = true_function(p, θ)
    return ŷ .- y_target
end

θ_init = θ_true .+ randn!(similar(θ_true)) * 0.1
prob_oop = NonlinearLeastSquaresProblem{false}(loss_function, θ_init, x)

solver = LevenbergMarquardt()

@time sol = solve(prob, solver; maxiters = 10000, abstol = 1e-8)

optf = OptimizationFunction(prob_oop.f, AutoForwardDiff())
optprob = OptimizationProblem(optf, prob_oop.u0, prob_oop.p)
@time sol = solve(optprob, NLopt.LD_LBFGS(); maxiters = 10000, abstol = 1e-8)

optprob = OptimizationProblem(prob_oop, AutoForwardDiff())
@time sol = solve(optprob, NLopt.LD_LBFGS(); maxiters = 10000, abstol = 1e-8)
