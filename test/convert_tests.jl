using Test, SciMLBase

@testset "Convert NonlinearFunction to ODEFunction" begin
    f! = NonlinearFunction((du, u, p) -> du[1] = u[1] - p[1] + p[2])
    f = NonlinearFunction((u, p) -> u .- p[1] .+ p[2])

    _f! = convert(ODEFunction, f!)
    _f = convert(ODEFunction, f)

    @test _f! isa ODEFunction && isinplace(_f!)
    @test _f isa ODEFunction && !isinplace(_f)
end
