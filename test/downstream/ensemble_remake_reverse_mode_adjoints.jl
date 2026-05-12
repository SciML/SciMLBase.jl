using SciMLBase, OrdinaryDiffEq, Test
using SciMLBase: EnsembleProblem, EnsembleSerial, EnsembleSolution
using Zygote, ForwardDiff
import ChainRulesCore

@testset "EnsembleSolution constructor pulls NamedTuple cotangent" begin
    f(u, p, t) = -u
    prob = ODEProblem(f, [1.0], (0.0, 1.0))
    sols = [solve(prob, Tsit5(); saveat = 0.5) for _ in 1:3]
    arrarr = [[copy(s.u[j]) for j in eachindex(s.u)] for s in sols]

    _, back = Zygote.pullback(EnsembleSolution, sols, 0.0, true, nothing)
    sim_cot, t_cot, c_cot, s_cot = back((u = arrarr,))
    @test sim_cot == arrarr
    @test t_cot === nothing && c_cot === nothing && s_cot === nothing

    _, pb = ChainRulesCore.rrule(EnsembleSolution, sols, 0.0, true, nothing)
    cot = pb((u = arrarr,))
    @test cot[2] == arrarr
    @test cot[1] === ChainRulesCore.NoTangent()
    @test all(cot[i] === ChainRulesCore.NoTangent() for i in 3:5)

    cot_t = pb(ChainRulesCore.Tangent{Any}(; u = arrarr))
    @test cot_t[2] == arrarr
end

@testset "remake(::ODEProblem; u0) gradient parity" begin
    f(u, p, t) = u
    base_prob = ODEProblem(f, [0.0, 0.0], (0.0, 1.0), [1.0])
    loss(p) = (q = remake(base_prob, u0 = [p[1] * 2, p[1] + 5]); sum(abs2, q.u0))
    p0 = [3.0]
    @test Zygote.gradient(loss, p0)[1] ≈ ForwardDiff.gradient(loss, p0) rtol=1e-6
end

@testset "remake(::ODEProblem; p) gradient parity" begin
    f(u, p, t) = p[1] * u
    base_prob = ODEProblem(f, [1.0], (0.0, 1.0), [0.5])
    loss(p) = (q = remake(base_prob, p = [p[1] * 3]); sum(abs2, q.p))
    p0 = [2.0]
    @test Zygote.gradient(loss, p0)[1] ≈ ForwardDiff.gradient(loss, p0) rtol=1e-6
end

@testset "remake field-pass-through gradient parity" begin
    f(u, p, t) = u
    base_prob = ODEProblem(f, [1.0], (0.0, 1.0), [1.0])
    loss(p) = (q = remake(base_prob, u0 = [p[1]]); sum(abs2, q.u0))
    p0 = [2.5]
    @test Zygote.gradient(loss, p0)[1] ≈ ForwardDiff.gradient(loss, p0) rtol=1e-6
end

@testset "_remake_ode_inner rrule cotangent distribution" begin
    base_prob = ODEProblem((u, p, t) -> u, [1.0, 2.0], (0.0, 1.0), [3.0])
    Δ_u0 = [10.0, 20.0]
    Δ = ChainRulesCore.Tangent{Any}(;
        f = ChainRulesCore.NoTangent(),
        u0 = Δ_u0,
        tspan = ChainRulesCore.NoTangent(),
        p = ChainRulesCore.NoTangent(),
        kwargs = ChainRulesCore.NoTangent(),
        problem_type = ChainRulesCore.NoTangent(),
    )

    # u0 supplied → cotangent flows to the u0 positional.
    _, pb = ChainRulesCore.rrule(
        SciMLBase._remake_ode_inner,
        base_prob, missing, [9.9, 8.8], missing, missing, missing,
        true, Val{true}, false, nothing, NamedTuple()
    )
    cot = pb(Δ)
    @test length(cot) == 12
    @test cot[1] === ChainRulesCore.NoTangent()
    @test cot[4] == Δ_u0
    @test cot[2].u0 === nothing
    @test all(cot[i] === ChainRulesCore.NoTangent() for i in 7:12)

    # u0 not supplied → cotangent accumulates onto prob.u0.
    _, pb2 = ChainRulesCore.rrule(
        SciMLBase._remake_ode_inner,
        base_prob, missing, missing, missing, [99.0], missing,
        true, Val{true}, false, nothing, NamedTuple()
    )
    cot2 = pb2(Δ)
    @test cot2[4] === ChainRulesCore.NoTangent()
    @test cot2[2].u0 == Δ_u0
end
