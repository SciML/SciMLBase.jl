using SciMLBase
using LinearAlgebra

@testset "DiffEqOperator" begin
    A = rand(10,10);
    @test eachindex(A) === eachindex(SciMLBase.DiffEqArrayOperator(A))
    @test eachindex(A') === eachindex(SciMLBase.DiffEqArrayOperator(A'))

    A = Matrix(I,10,10) |> SciMLBase.DiffEqArrayOperator
    @test factorize(A) isa SciMLBase.FactorizedDiffEqArrayOperator
end
