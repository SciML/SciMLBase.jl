using SciMLBase, SciMLOperators
using LinearAlgebra

@testset "DiffEqOperator" begin
    n = 10
    u = rand(n)
    p = nothing
    t = 0

    A  = rand(n,n)
    At = A'

    AA  = SciMLBase.SciMLOperators.MatrixOperator(A)
    AAt = AA'

    @test AA  isa SciMLBase.SciMLOperators.MatrixOperator
    @test AAt isa SciMLBase.SciMLOperators.MatrixOperator

    FF  = factorize(AA)
    FFt = FF'

    @test FF  isa SciMLBase.SciMLOperators.InverseOperator
    @test FFt isa SciMLBase.SciMLOperators.InverseOperator

    @test eachindex(A)  === eachindex(AA)
    @test eachindex(A') === eachindex(AAt) === eachindex(SciMLOperators.MatrixOperator(At))

    @test A  ≈ convert(AbstractMatrix, AA ) ≈ convert(AbstractMatrix, FF )
    @test At ≈ convert(AbstractMatrix, AAt) ≈ convert(AbstractMatrix, FFt)

    @test A  ≈ Matrix(AA ) ≈ Matrix(FF )
    @test At ≈ Matrix(AAt) ≈ Matrix(FFt)

    @test A  * u ≈ AA(u,p,t)  ≈ FF(u,p,t)
    @test At * u ≈ AAt(u,p,t) ≈ FFt(u,p,t)

    @test A  \ u ≈ AA  \ u ≈ FF  \ u
    @test At \ u ≈ AAt \ u ≈ FFt \ u
end
