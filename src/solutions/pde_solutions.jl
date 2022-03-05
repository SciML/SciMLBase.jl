"""
$(TYPEDEF)
"""
struct PDESolution{PDEType, DiscretizerType, NumProblemType, NumSolutionType}
    pde_system::PDEType
    discretizer::DiscretizerType
    num_problem::NumProblemType
    num_solution::NumSolutionType
end
# PDESolution(pdesys,discretizer,numprob,numsol) 
