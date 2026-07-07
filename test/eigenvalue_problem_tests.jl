using SciMLBase
using Test

struct DummyEigenAlg end

A = [2.0 0.0; 0.0 3.0]

@testset "Construction defaults" begin
    prob = EigenvalueProblem(A)
    @test prob.A === A
    @test prob.B === nothing
    @test prob.num_eigenpairs === nothing
    @test prob.eigentarget === EigenvalueTarget.LargestMagnitude
    @test prob.shift === nothing
end

@testset "eigentarget accepts EigenvalueTarget values" begin
    @test EigenvalueProblem(A; eigentarget = EigenvalueTarget.LargestMagnitude).eigentarget ===
        EigenvalueTarget.LargestMagnitude
    @test EigenvalueProblem(A; eigentarget = EigenvalueTarget.SmallestMagnitude).eigentarget ===
        EigenvalueTarget.SmallestMagnitude
    @test EigenvalueProblem(A; eigentarget = EigenvalueTarget.LargestRealPart).eigentarget ===
        EigenvalueTarget.LargestRealPart
    @test EigenvalueProblem(A; eigentarget = EigenvalueTarget.SmallestRealPart).eigentarget ===
        EigenvalueTarget.SmallestRealPart
    @test EigenvalueProblem(A; eigentarget = EigenvalueTarget.LargestImaginaryPart).eigentarget ===
        EigenvalueTarget.LargestImaginaryPart
    @test EigenvalueProblem(A; eigentarget = EigenvalueTarget.SmallestImaginaryPart).eigentarget ===
        EigenvalueTarget.SmallestImaginaryPart
end

@testset "Invalid `eigentarget` errors at construction" begin
    @test_throws TypeError EigenvalueProblem(A; eigentarget = :LM)
    @test_throws TypeError EigenvalueProblem(A; eigentarget = 5)
end

@testset "Generalized problem stores B" begin
    B = [1.0 0.0; 0.0 1.0]
    prob = EigenvalueProblem(A, B; num_eigenpairs = 1)
    @test prob.B === B
    @test prob.num_eigenpairs == 1
end

@testset "build_eigenvalue_solution" begin
    prob = EigenvalueProblem(A)
    values = [2.0, 3.0]
    vectors = [1.0 0.0; 0.0 1.0]
    sol = SciMLBase.build_eigenvalue_solution(prob, DummyEigenAlg(), values, vectors)
    @test sol.u == values
    @test sol.vectors == vectors
    @test sol.retcode === ReturnCode.Success
end
