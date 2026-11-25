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

# Derivative fields follow the λ-extended convention: jac(u, p, λ) / jac(J, u, p, λ).
# `lambda_extended = true` makes the NonlinearFunction constructor validate jac/jvp/vjp
# against these arities so a bare λ-extended jac (no λ-free dummy method) is accepted.
j_oop(u, p, λ) = reshape([(1 - λ) + λ * 2 * u[1];;], 1, 1)
j_iip(J, u, p, λ) = (J[1, 1] = (1 - λ) + λ * 2 * u[1]; nothing)

@testset "λ-extended jac: bare 3-arg oop / 4-arg iip construct with lambda_extended" begin
    # without the opt-in, the standard conformance check rejects them (unchanged)
    @test_throws SciMLBase.NonconformingFunctionsError NonlinearFunction{false}(
        f_oop; jac = j_oop
    )
    @test_throws SciMLBase.TooManyArgumentsError NonlinearFunction{true}(
        f_iip; jac = j_iip
    )

    nf_oop = NonlinearFunction{false}(f_oop; jac = j_oop, lambda_extended = true)
    @test SciMLBase.__has_jac(nf_oop)
    @test nf_oop.jac === j_oop
    prob_oop = HomotopyProblem(nf_oop, u0, p)
    @test SciMLBase.isinplace(prob_oop) == false
    @test prob_oop.f.jac === j_oop

    nf_iip = NonlinearFunction{true}(f_iip; jac = j_iip, lambda_extended = true)
    @test nf_iip.jac === j_iip
    prob_iip = HomotopyProblem(nf_iip, u0, p)
    @test SciMLBase.isinplace(prob_iip) == true
    @test prob_iip.f.jac === j_iip
end

@testset "λ-extended autodetect: NonlinearFunction(f; lambda_extended = true)" begin
    nf_oop = NonlinearFunction(f_oop; jac = j_oop, lambda_extended = true)
    @test SciMLBase.isinplace(nf_oop) == false
    nf_iip = NonlinearFunction(f_iip; jac = j_iip, lambda_extended = true)
    @test SciMLBase.isinplace(nf_iip) == true
end

@testset "λ-extended jvp/vjp arities shift too" begin
    jvp_oop(v, u, p, λ) = ((1 - λ) + λ * 2 * u[1]) .* v
    vjp_iip(vJ, v, u, p, λ) = (vJ[1] = ((1 - λ) + λ * 2 * u[1]) * v[1]; nothing)
    nf_oop = NonlinearFunction{false}(f_oop; jvp = jvp_oop, lambda_extended = true)
    @test nf_oop.jvp === jvp_oop
    nf_iip = NonlinearFunction{true}(f_iip; vjp = vjp_iip, lambda_extended = true)
    @test nf_iip.vjp === vjp_iip
end

@testset "lambda_extended still rejects nonconforming derivative functions" begin
    # an iip λ-extended jac with an oop residual is still nonconforming
    @test_throws SciMLBase.NonconformingFunctionsError NonlinearFunction{false}(
        f_oop; jac = j_iip, lambda_extended = true
    )
    j_toomany(J, u, p, λ, x) = nothing
    @test_throws SciMLBase.TooManyArgumentsError NonlinearFunction{true}(
        f_iip; jac = j_toomany, lambda_extended = true
    )
end

@testset "λ-extended structure fields pass through" begin
    import LinearAlgebra
    proto = LinearAlgebra.Diagonal(ones(1))
    nf = NonlinearFunction{true}(
        f_iip; jac = j_iip, jac_prototype = proto, colorvec = [1],
        lambda_extended = true
    )
    prob = HomotopyProblem(nf, u0, p)
    @test prob.f.jac_prototype === proto
    @test prob.f.sparsity === proto            # sparsity defaults to jac_prototype
    @test prob.f.colorvec == [1]
end
