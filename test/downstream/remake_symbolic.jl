using ModelingToolkit

@testset "symbolic remake with nested system" begin
    function makesys(name)
        @parameters t a=1.0
        @variables x(t) = 0.0
        D = Differential(t)
        ODESystem([D(x) ~ -a * x]; name)
    end

    function makecombinedsys()
        sys1 = makesys(:sys1)
        sys2 = makesys(:sys2)
        @parameters t b=1.0
        ODESystem(Equation[], t, [], [b]; systems = [sys1, sys2], name = :foo)
    end

    sys = makecombinedsys()
    @unpack sys1, b = sys
    prob = ODEProblem(sys, Pair[])

    prob_new = remake(prob, p = Dict(sys1.a => 3.0, b => 4.0), u0 = Dict(sys1.x => 1.0))
    @test prob_new.p == [4.0, 3.0, 1.0]
    @test prob_new.u0 == [1.0, 0.0]

    @test_throws ArgumentError remake(prob, p = [1.0])
    @test_throws ArgumentError remake(prob, u0 = [1.0])
end
