using SciMLBase
using LinearAlgebra

@testset "DiffEqOperator" begin
    n = 10
    u = rand(n)

    A  = rand(n,n)
    At = A'

    AA  = SciMLBase.DiffEqArrayOperator(A)
    AAt = AA'

    FF  = factorize(AA)
    FFt = FF'

    p = nothing
    t = 0

    @test eachindex(A)  === eachindex(AA)
    @test eachindex(A') === eachindex(AAt) === eachindex(DiffEqArrayOperator(At))

    @test A  * u ≈ AA(u,p,t)  ≈ FF(u,p,t)
    @test At * u ≈ AAt(u,p,t) ≈ FFt(u,p,t)

    @test A  \ u ≈ AA  \ u ≈ FF  \ u
    @test At \ u ≈ AAt \ u ≈ FFt \ u

    A = Matrix(I,n,n) |> SciMLBase.DiffEqArrayOperator
    @test factorize(A) isa SciMLBase.FactorizedDiffEqArrayOperator
end
