using SciMLBase
using LinearAlgebra

@testset "DiffEqOperator" begin
    n = 10
    u = rand(n)

    A  = rand(n,n)
    At = A'

    AA  = SciMLBase.DiffEqArrayOperator(A)
    AAt = SciMLBase.DiffEqArrayOperator(A')

    FF  = factorize(AA)
    FFt = FF'

    p = nothing
    t = 0

    @test eachindex(A) === eachindex(AA)
    @test eachindex(A') === eachindex(AAt)

    @test A  * u ≈ AA(u,p,t)
    @test At * u ≈ AAt(u,p,t)

    AAt = AA'
    @test eachindex(At) === eachindex(AAt)
    @test At * u ≈ AAt(u,p,t)

    A = Matrix(I,n,n) |> SciMLBase.DiffEqArrayOperator
    @test factorize(A) isa SciMLBase.FactorizedDiffEqArrayOperator
end
