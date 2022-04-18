using SciMLBase
using LinearAlgebra

@testset "DiffEqOperator" begin
    n = 10
    A = rand(n,n)
    u = rand(n)

    AA  = SciMLBase.DiffEqArrayOperator(A)
    AAt = SciMLBase.DiffEqArrayOperator(A')
    @test eachindex(A) === eachindex(AA)
    @test eachindex(A') === eachindex(AAt)

    @test A * u  ≈ AA  * u
    @test A' * u ≈ AAt * u


    @test has_adjoint(AA) === true

    AAt = AA'
    @test eachindex(A') === eachindex(AAt)
    @test A' * u ≈ AAt * u

    update_coefficients!(AAt,nothing, nothing, nothing)
    @test eachindex(A') === eachindex(AAt)
    @test A' * u ≈ AAt * u

    A = Matrix(I,n,n) |> SciMLBase.DiffEqArrayOperator
    @test factorize(A) isa SciMLBase.FactorizedDiffEqArrayOperator
end
