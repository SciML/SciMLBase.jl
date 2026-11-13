using SciMLBase, ForwardDiff, Test
using ForwardDiff: Dual, Tag

# https://github.com/SciML/SciMLBase.jl/issues/1664
@testset "value/unitfulvalue strip all layers" begin
    D1 = Dual{typeof(Tag(identity, Float64)), Float64, 1}
    D2 = Dual{typeof(Tag(abs, D1)), D1, 1}
    D3 = Dual{typeof(Tag(sin, D2)), D2, 2}

    @test SciMLBase.value(Float64) === Float64
    @test SciMLBase.unitfulvalue(Float64) === Float64

    for f in (SciMLBase.value, SciMLBase.unitfulvalue), D in (D1, D2, D3)
        @test f(D) === Float64
        @test f(one(D)) === 1.0
    end
end
