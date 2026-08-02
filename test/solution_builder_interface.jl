using SciMLBase
using Test

module ExternalSolutionBuilderContract
    using SciMLBase:
        AbstractLinearAlgorithm, EigenvalueProblem, ReturnCode, build_eigenvalue_solution,
        build_linear_solution

    struct ExternalLinearAlgorithm <: AbstractLinearAlgorithm end

    const LINEAR_RESULT = build_linear_solution(
        ExternalLinearAlgorithm(), [1.0, 2.0], Ref(0.0), nothing;
        retcode = ReturnCode.Success, iters = 3, stats = (matvecs = 4,)
    )

    const EIGEN_PROBLEM = EigenvalueProblem([2.0 0.0; 0.0 3.0])
    const EIGEN_RESULT = build_eigenvalue_solution(
        EIGEN_PROBLEM, ExternalLinearAlgorithm(), [2.0, 3.0], [1.0 0.0; 0.0 1.0];
        resid = [0.0, 0.0], stats = (factorizations = 1,)
    )
end

@testset "Public solution builder interface" begin
    if VERSION >= v"1.11"
        @test Base.ispublic(SciMLBase, :build_linear_solution)
        @test Base.ispublic(SciMLBase, :build_eigenvalue_solution)
    end

    linear = ExternalSolutionBuilderContract.LINEAR_RESULT
    @test linear.u == [1.0, 2.0]
    @test linear.resid[] == 0.0
    @test linear.alg isa ExternalSolutionBuilderContract.ExternalLinearAlgorithm
    @test linear.retcode === SciMLBase.ReturnCode.Success
    @test linear.iters == 3
    @test linear.cache === nothing
    @test linear.stats == (matvecs = 4,)

    eigen = ExternalSolutionBuilderContract.EIGEN_RESULT
    @test eigen.u == [2.0, 3.0]
    @test eigen.vectors == [1.0 0.0; 0.0 1.0]
    @test eigen.prob === ExternalSolutionBuilderContract.EIGEN_PROBLEM
    @test eigen.alg isa ExternalSolutionBuilderContract.ExternalLinearAlgorithm
    @test eigen.retcode === SciMLBase.ReturnCode.Success
    @test eigen.resid == [0.0, 0.0]
    @test eigen.stats == (factorizations = 1,)
end
