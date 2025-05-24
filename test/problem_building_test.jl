using Test, SciMLBase, SymbolicIndexingInterface, Accessors, StaticArrays

function simplependulum!(du, u, p, t)
    θ = u[1]
    dθ = u[2]
    du[1] = dθ
    du[2] = -9.81 * sin(θ)
end
function bc!(residual, u, p, t)
    residual[1] = u[1][1] + pi / 2
    residual[2] = u[end][1] - pi / 2
end
prob_bvp = BVProblem(simplependulum!, bc!, [pi / 2, pi / 2], (0, 1.0))
@test prob_bvp.tspan === (0.0, 1.0)

@testset "`constructorof` tests" begin
    probs = []

    function lorenz!(du, u, p, t)
        du[1] = p[1] * (u[2] - u[1])
        du[2] = u[1] * (p[2] - u[3]) - u[2]
        du[3] = u[1] * u[2] - p[3] * u[3]
    end
    u0 = [1.0; 2.0; 3.0]
    du0 = similar(u0)
    p = [10.0, 20.0, 30.0]
    tspan = (0.0, 100.0)
    lorenz!(du0, u0, p, tspan[1])
    sys = SymbolCache([:x, :y, :z], [:a, :b, :c], :t)
    fn = ODEFunction(lorenz!; sys)
    push!(probs, ODEProblem(fn, u0, tspan, p))

    function daelorenz!(resid, du, u, p, t)
        lorenz!(resid, u, p, t)
        resid .-= du
    end
    fn = DAEFunction(daelorenz!; sys)
    push!(probs, DAEProblem(fn, du0, u0, tspan, p))

    function ddelorenz!(du, u, h, p, t)
        du[1] = p[1] * (u[2] - u[1])
        du[2] = u[1] * (p[2] - u[3]) - u[2]
        du[3] = u[1] * u[2] - p[3] * u[3]
    end

    function history(p, t)
        return u0 .- t
    end

    fn = DDEFunction(ddelorenz!; sys)
    push!(probs, DDEProblem(fn, u0, history, tspan, p))

    function noise!(du, u, p, t)
        du .= 0.1u
    end
    fn = SDEFunction(lorenz!, noise!; sys)
    push!(probs, SDEProblem(fn, u0, tspan, p))

    fn = SDDEFunction(ddelorenz!, noise!; sys)
    push!(probs, SDDEProblem(fn, noise!, u0, history, tspan, p))

    function nllorenz!(du, u, p)
        lorenz!(du, u, p, 0.0)
    end

    fn = NonlinearFunction(nllorenz!; sys)
    push!(probs, NonlinearProblem(fn, u0, p))
    push!(probs, NonlinearLeastSquaresProblem(fn, u0, p))

    @testset "$(SciMLBase.parameterless_type(typeof(prob)))" for prob in probs
        newprob = @reset prob.u0 = u0 .+ 1
        @test typeof(newprob) == typeof(prob)
    end
end

# test for tspan promotion in DiscreteProblem
let
    p = (0.1 / 1000, 0.01)
    u₀ = [1.0]
    tspan = (0.0, 1.0)
    dprob = DiscreteProblem(u₀, tspan, p)
    @test dprob.tspan === (0.0, 1.0)

    dprob2 = DiscreteProblem(u₀, 1.0, p)
    @test dprob.tspan === (0.0, 1.0)

    dprob3 = DiscreteProblem(u₀, [0.0, 1.0], p)
    @test dprob.tspan === (0.0, 1.0)

    dprob4 = DiscreteProblem{true}(nothing, nothing, p)
    dprob4b = DiscreteProblem{true}(nothing, nothing)
    dprob5 = DiscreteProblem{true}(SciMLBase.DISCRETE_INPLACE_DEFAULT, u₀, tspan, p)
    dprob6 = DiscreteProblem{true}(SciMLBase.DISCRETE_INPLACE_DEFAULT, u₀, tspan)
    dprob7 = DiscreteProblem{true}((du, u, p, t) -> du[1] = 0, u₀, tspan)
    @test dprob7.u0 === u₀
    @test dprob7.tspan === tspan
    dprob8 = DiscreteProblem{true}(nothing, u₀, tspan)
    @test dprob8.u0 === u₀
    @test dprob8.tspan === tspan
end

@testset "SCCNonlinearProblem with static arrays" begin
    function f1(u, p)
        y = u[1]
        x = p[1]
        return SA[1 - y ^ 2 - x ^ 2]
    end

    function f2(u, p)
        yt = u[1]
        x, xt, y = p
        return SA[-2y * yt - 2x * xt]
    end

    function f3(u, p)
        lam = u[1]
        x, xt, y, yt = p
        return SA[-2xt ^ 2 - 2yt ^ 2 - 2y * (-1 + y * lam) - 2x ^ 2 * lam]
    end

    explicit1 = Returns(nothing)
    function explicit2(p, sols)
        p[3] = sols[1].u[1]
    end
    function explicit3(p, sols)
        p[4] = sols[2].u[1]
    end

    p = [1.0, 0.0, NaN, NaN]
    prob1 = NonlinearProblem(f1, SA[1.0], p)
    prob2 = NonlinearProblem(f2, SA[0.0], p)
    prob3 = NonlinearProblem(f3, SA[1.0], p)
    sccprob = SCCNonlinearProblem(
        (prob1, prob2, prob3), (explicit1, explicit2, explicit3), p, true)

    @test !SciMLBase.isinplace(sccprob)
    @test sccprob isa SCCNonlinearProblem{SVector{3, Float64}}
    @test state_values(sccprob) isa SVector{3, Float64}
    @test sccprob.p === prob1.p === prob2.p === prob3.p

    sccprob2 = @inferred remake(sccprob; u0 = SA[2.0, 1.0, 2.0])
    @test !SciMLBase.isinplace(sccprob2)
    @test sccprob2 isa SCCNonlinearProblem{SVector{3, Float64}}
    @test state_values(sccprob2) isa SVector{3, Float64}
end
