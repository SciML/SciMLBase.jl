using SciMLBase
using Test

# out-of-place residual f(u, p, λ): λ is a separate scalar argument, p is untouched.
# Linear-to-quadratic family: λ=0 root u=p[1], λ=1 root u=√p[1].
f_oop(u, p, λ) = [(1 - λ) * (u[1] - p[1]) + λ * (u[1]^2 - p[1])]
u0 = [2.0]
p = [2.0]

@testset "HomotopyProblem construction (explicit NonlinearFunction, oop)" begin
    nf = NonlinearFunction{false}(f_oop)
    prob = SciMLBase.HomotopyProblem{false}(nf, u0, p)
    @test prob isa SciMLBase.HomotopyProblem
    @test prob isa SciMLBase.AbstractNonlinearProblem
    @test prob.f === nf
    @test prob.u0 == u0
    @test prob.p == p
    @test prob.λspan == (0.0, 1.0)
    @test SciMLBase.isinplace(prob) == false
end

f_iip(du, u, p, λ) = (du[1] = (1 - λ) * (u[1] - p[1]) + λ * (u[1]^2 - p[1]); nothing)

@testset "HomotopyProblem outer constructors (raw function, oop + iip autodetect)" begin
    # raw out-of-place f(u, p, λ): 3 args ⇒ isinplace(f, 4) detects false
    prob_oop = SciMLBase.HomotopyProblem(f_oop, u0, p)
    @test SciMLBase.isinplace(prob_oop) == false
    @test prob_oop.f isa SciMLBase.NonlinearFunction

    # raw in-place f(du, u, p, λ): 4 args ⇒ isinplace(f, 4) detects true
    prob_iip = SciMLBase.HomotopyProblem(f_iip, u0, p)
    @test SciMLBase.isinplace(prob_iip) == true

    # p needs no particular structure anymore: NamedTuple works
    prob_nt = SciMLBase.HomotopyProblem(
        (u, p, λ) -> [(1 - λ) * (u[1] - p.c) + λ * (u[1]^2 - p.c)], u0, (c = 2.0,)
    )
    @test prob_nt.p == (c = 2.0,)

    # explicit AbstractNonlinearFunction, no {iip} given
    nf = NonlinearFunction{false}(f_oop)
    prob2 = SciMLBase.HomotopyProblem(nf, u0, p; λspan = (0.0, 2.0))
    @test prob2.λspan == (0.0, 2.0)
end

using Accessors: @reset

@testset "HomotopyProblem ConstructionBase / @reset preserves type" begin
    prob = SciMLBase.HomotopyProblem(f_oop, u0, p)
    newprob = @reset prob.u0 = u0 .+ 1
    @test typeof(newprob) == typeof(prob)
    @test newprob.u0 == u0 .+ 1
    @test newprob.λspan == (0.0, 1.0)
end

import SymbolicIndexingInterface as SII

@testset "HomotopyProblem export + inherited SII traits" begin
    @test isdefined(SciMLBase, :HomotopyProblem)
    @test :HomotopyProblem in names(SciMLBase)

    prob = HomotopyProblem(f_oop, u0, p)
    @test SII.parameter_values(prob) == p
    @test SII.state_values(prob) == u0
end

@testset "HomotopyProblem remake" begin
    prob = HomotopyProblem(f_oop, u0, p)
    rp = SciMLBase.remake(prob; u0 = [5.0], p = [3.0])
    @test rp isa HomotopyProblem
    @test rp.u0 == [5.0]
    @test rp.p == [3.0]
    @test rp.λspan == (0.0, 1.0)              # preserved
    @test SciMLBase.isinplace(rp) == false

    rp2 = SciMLBase.remake(prob; λspan = (0.0, 2.0))
    @test rp2.λspan == (0.0, 2.0)
    @test rp2.u0 == u0                        # unchanged
end

@testset "HomotopyProblem @reset to a raw f exercises constructorof else-branch" begin
    # a plain 3-arg function f(u, p, λ) — not an AbstractNonlinearFunction —
    # forces the else-branch of constructorof: isinplace(f, 4) ⇒ out-of-place.
    f_raw(u, p, λ) = [(1 - λ) * (u[1] - p[1]) + λ * (u[1]^2 - p[1])]
    prob = HomotopyProblem(f_oop, u0, p)
    newprob = @reset prob.f = f_raw
    @test newprob isa HomotopyProblem
    @test newprob.f isa SciMLBase.NonlinearFunction       # raw f got re-wrapped
    @test SciMLBase.isinplace(newprob) == false
    @test newprob.λspan == (0.0, 1.0)                     # preserved
end
