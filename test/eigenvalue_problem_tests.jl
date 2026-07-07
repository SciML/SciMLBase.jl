using SciMLBase
using Test

struct DummyEigenAlg end

A = [2.0 0.0; 0.0 3.0]

@testset "Construction defaults" begin
    prob = EigenvalueProblem(A)
    @test prob.A === A
    @test prob.B === nothing
    @test prob.nev === nothing
    @test prob.which === EigenvalueTarget.LargestMagnitude
    @test prob.sigma === nothing
end

@testset "Symbol `which` normalizes to EigenvalueTarget" begin
    @test EigenvalueProblem(A; which = :LM).which === EigenvalueTarget.LargestMagnitude
    @test EigenvalueProblem(A; which = :SM).which === EigenvalueTarget.SmallestMagnitude
    @test EigenvalueProblem(A; which = :LR).which === EigenvalueTarget.LargestRealPart
    @test EigenvalueProblem(A; which = :SR).which === EigenvalueTarget.SmallestRealPart
    @test EigenvalueProblem(A; which = :LI).which === EigenvalueTarget.LargestImaginaryPart
    @test EigenvalueProblem(A; which = :SI).which === EigenvalueTarget.SmallestImaginaryPart
    @test EigenvalueProblem(A; which = EigenvalueTarget.SmallestRealPart).which ===
        EigenvalueTarget.SmallestRealPart
end

@testset "Invalid `which` errors at construction" begin
    @test_throws ArgumentError EigenvalueProblem(A; which = :XX)
    @test_throws ArgumentError EigenvalueProblem(A; which = 5)
end

@testset "Generalized problem stores B" begin
    B = [1.0 0.0; 0.0 1.0]
    prob = EigenvalueProblem(A, B; nev = 1)
    @test prob.B === B
    @test prob.nev == 1
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
