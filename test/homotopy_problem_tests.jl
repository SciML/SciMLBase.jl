using SciMLBase
using Test

# out-of-place residual H(u, p): p = [a, λ]; here a trivial linear family.
f_oop(u, p) = [u[1] - p[1] * p[2]]
u0 = [0.0]
p = [2.0, 1.0]          # p[2] is λ

@testset "HomotopyProblem construction (explicit NonlinearFunction, oop)" begin
    nf = NonlinearFunction{false}(f_oop)
    prob = SciMLBase.HomotopyProblem{false}(nf, u0, p; homotopy_parameter = 2)
    @test prob isa SciMLBase.HomotopyProblem
    @test prob isa SciMLBase.AbstractNonlinearProblem
    @test prob.f === nf
    @test prob.u0 == u0
    @test prob.p == p
    @test prob.homotopy_parameter == 2
    @test prob.λspan == (0.0, 1.0)
    @test SciMLBase.isinplace(prob) == false
end

f_iip(du, u, p) = (du[1] = u[1] - p[1] * p[2]; nothing)

@testset "HomotopyProblem outer constructors (raw function, oop + iip autodetect)" begin
    # raw out-of-place function: isinplace must autodetect false
    prob_oop = SciMLBase.HomotopyProblem(f_oop, u0, p; homotopy_parameter = 2)
    @test SciMLBase.isinplace(prob_oop) == false
    @test prob_oop.f isa SciMLBase.NonlinearFunction

    # raw in-place function: isinplace must autodetect true
    prob_iip = SciMLBase.HomotopyProblem(f_iip, u0, p; homotopy_parameter = 2)
    @test SciMLBase.isinplace(prob_iip) == true

    # explicit AbstractNonlinearFunction, no {iip} given
    nf = NonlinearFunction{false}(f_oop)
    prob2 = SciMLBase.HomotopyProblem(nf, u0, p; homotopy_parameter = 2, λspan = (0.0, 2.0))
    @test prob2.λspan == (0.0, 2.0)
end

using Accessors: @reset

@testset "HomotopyProblem ConstructionBase / @reset preserves type" begin
    prob = SciMLBase.HomotopyProblem(f_oop, u0, p; homotopy_parameter = 2)
    newprob = @reset prob.u0 = u0 .+ 1
    @test typeof(newprob) == typeof(prob)
    @test newprob.u0 == u0 .+ 1
    @test newprob.homotopy_parameter == 2
    @test newprob.λspan == (0.0, 1.0)
end
