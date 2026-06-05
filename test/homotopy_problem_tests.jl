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
