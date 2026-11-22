using Test, SciMLBase
using Mooncake: Mooncake

@testset "Mooncake map with mixed float precisions" begin
    x = Union{Float32, Float64}[1.0f0, 1.0 + 2.0^-35]
    for mapper in (SciMLBase.responsible_map, SciMLBase.tmap)
        @testset "$(nameof(mapper))" begin
            expected = mapper(identity, x)
            x_dual = Mooncake.zero_fcodual(x)
            y_dual, pullback = Mooncake.rrule!!(
                Mooncake.zero_fcodual(mapper), Mooncake.zero_fcodual(identity), x_dual
            )
            actual = Mooncake.primal(y_dual)
            @test typeof(actual) === typeof(expected)
            @test all(actual[i] === x[i] for i in eachindex(x))

            seed = Union{Float32, Float64}[2.0f0, 3.0 + 2.0^-35]
            Mooncake.tangent(y_dual) .= seed
            pullback(Mooncake.NoRData())
            dx = Mooncake.tangent(x_dual)
            @test all(dx[i] === seed[i] for i in eachindex(x))
        end
    end
end
