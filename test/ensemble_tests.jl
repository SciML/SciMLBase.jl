using Test, SciMLBase, SciMLBase.EnsembleAnalysis

EA = SciMLBase.EnsembleAnalysis

# tests for https://github.com/SciML/DifferentialEquations.jl/issues/731
# make sure integer inputs work
@test EA.componentwise_mean([1,1,1]) ≈ 1.0
m,v = EA.componentwise_meanvar([1,2,3]) 
@test m ≈ 2.0
@test v ≈ 1.0
mx,my,C = EA.componentwise_meancov([1,2,3], [3,2,1])
@test mx ≈ 2.0
@test my ≈ 2.0
@test C ≈ -1.0
mx,my,C = EA.componentwise_weighted_meancov([1,2,3], [3,2,1], [1,2,1])
@test mx ≈ 2.0
@test my ≈ 2.0
@test C ≈ -.8
