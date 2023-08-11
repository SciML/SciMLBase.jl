using Test, SciMLBase, SciMLBase.EnsembleAnalysis

EA = SciMLBase.EnsembleAnalysis

# tests for https://github.com/SciML/DifferentialEquations.jl/issues/731
# make sure integer inputs work
@test all(EA.componentwise_mean([[1, 1], [2, 2], [3, 3]]) .≈ [2.0, 2.0])
m, v = EA.componentwise_meanvar([[1, 1], [2, 2], [3, 3]])
@test all(m .≈ 2.0)
@test all(v .≈ 1.0)
mx, my, C = EA.componentwise_meancov([[1, 1], [2, 2], [3, 3]], [[3, 3], [2, 2], [1, 1]])
@test all(mx .≈ 2.0)
@test all(my .≈ 2.0)
@test all(C .≈ -1.0)
mx, my, C = EA.componentwise_weighted_meancov([[1, 1], [2, 2], [3, 3]],
    [[3, 3], [2, 2], [1, 1]],
    [[1, 1], [2, 2], [1, 1]])
@test all(mx .≈ 2.0)
@test all(my .≈ 2.0)
@test all(C .≈ -0.8)
