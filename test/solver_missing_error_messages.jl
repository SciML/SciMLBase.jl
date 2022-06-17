using SciMLBase, Test
prob = OptimizationProblem((x,p)->sum(x),zeros(2))
@test_throws SciMLBase.OptimizerMissingError solve(prob,nothing)