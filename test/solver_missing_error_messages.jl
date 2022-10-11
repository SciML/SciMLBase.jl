using SciMLBase, Test
prob = OptimizationProblem((x, p) -> sum(x), zeros(2))
@test_throws SciMLBase.OptimizerMissingError solve(prob, nothing)

struct OptAlg end

SciMLBase.callbacks_support(::OptAlg) = false
@test_throws SciMLBase.IncompatibleOptimizerError solve(prob, OptAlg(),
                                                        callback = (args...) -> false)

prob = OptimizationProblem((x, p) -> sum(x), zeros(2), lb = [-1.0, -1.0], ub = [1.0, 1.0])
@test_throws SciMLBase.IncompatibleOptimizerError solve(prob, OptAlg()) #by default isbounded is false

cons = (res, x, p) -> (res .= [x[1]^2 + x[2]^2])
optf = OptimizationFunction((x, p) -> sum(x), SciMLBase.NoAD(), cons = cons)
prob = OptimizationProblem(optf, zeros(2))
@test_throws SciMLBase.IncompatibleOptimizerError solve(prob, OptAlg()) #by default isconstrained is false

SciMLBase.isconstrained(::OptAlg) = true
optf = OptimizationFunction((x, p) -> sum(x), SciMLBase.NoAD())
prob = OptimizationProblem(optf, zeros(2))
@test_throws SciMLBase.IncompatibleOptimizerError solve(prob, OptAlg())
