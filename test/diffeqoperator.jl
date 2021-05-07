
@testset "DiffEqOperator" begin
    A = rand(10,10);
    @test eachindex(A) === eachindex(SciMLBase.DiffEqArrayOperator(A))
    @test eachindex(A') === eachindex(SciMLBase.DiffEqArrayOperator(A'))
end

