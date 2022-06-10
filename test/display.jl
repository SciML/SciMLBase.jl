using Test, SciMLBase


showtest(x, with_color=true) = sprint() do io
    Base.show(IOContext(io, :color => with_color), MIME"text/plain"(), x)
end


@testset "Respect :color setting in IOContext" begin
    local prob = ODEProblem((u,p,t)->u, 1.0, (0.0, 1.0), ones(1,1))

    # This test might be too specific
    # @test showtest(prob, true) == "\e[36mODEProblem\e[0m with uType \e[36mFloat64\e[0m and tType \e[36mFloat64\e[0m. In-place: \e[36mfalse\e[0m\ntimespan: (0.0, 1.0)\nu0: 1.0"


    # Only test whether the colorizers are printed:
    @test occursin(SciMLBase.TYPE_COLOR, showtest(prob, true))
    @test occursin(SciMLBase.NO_COLOR, showtest(prob, true))
    
    @test !occursin(SciMLBase.TYPE_COLOR, showtest(prob, false))
    @test !occursin(SciMLBase.NO_COLOR, showtest(prob, false))
end

