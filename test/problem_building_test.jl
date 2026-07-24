using Test, SciMLBase, SymbolicIndexingInterface, Accessors, StaticArrays

function simplependulum!(du, u, p, t)
    θ = u[1]
    dθ = u[2]
    du[1] = dθ
    return du[2] = -9.81 * sin(θ)
end
function bc!(residual, u, p, t)
    residual[1] = u[1][1] + pi / 2
    return residual[2] = u[end][1] - pi / 2
end
prob_bvp = BVProblem(simplependulum!, bc!, [pi / 2, pi / 2], (0, 1.0))
@test prob_bvp.tspan === (0.0, 1.0)

@testset "Structured ODE problem interfaces" begin
    steady_prob = SteadyStateProblem((u, p, t) -> p * u, 1.0, 2.0)
    @test current_time(steady_prob) === Inf

    incrementing_model! = (du, u, p, t, alpha = true, beta = false) ->
    (du .= alpha .* p .* u .+ beta .* du)
    incrementing_f = IncrementingODEFunction{true}(incrementing_model!)
    du = [10.0, 20.0]
    incrementing_f(du, [2.0, 4.0], -0.5, 0.0, 2.0, 1.0)
    @test du == [8.0, 16.0]

    incrementing_prob = IncrementingODEProblem(
        incrementing_f, [2.0, 4.0], (0.0, 1.0), -0.5
    )
    @test incrementing_prob isa ODEProblem
    @test incrementing_prob.f === incrementing_f
    @test incrementing_prob.problem_type isa IncrementingODEProblem{true}
    @test isinplace(incrementing_prob)

    specialized_incrementing_f = IncrementingODEFunction{
        true, SciMLBase.NoSpecialize,
    }(incrementing_model!)
    @test SciMLBase.specialization(specialized_incrementing_f) === SciMLBase.NoSpecialize
end

# Bare ODEProblem(SplitFunction, u0, tspan) must allocate `_func_cache` for iip,
# matching SplitODEProblem (otherwise f(du,u,p,t) hits ndims(::Type{Nothing})).
@testset "ODEProblem(SplitFunction) allocates _func_cache" begin
    f1! = (du, u, p, t) -> (du .= u)
    f2! = (du, u, p, t) -> (du .= -u)
    u0 = [1.0, 2.0]
    sf = SplitFunction(f1!, f2!)
    @test sf._func_cache === nothing
    @test_throws ArgumentError sf(similar(u0), u0, nothing, 0.0)

    prob = ODEProblem(sf, u0, (0.0, 1.0))
    @test prob.f._func_cache !== nothing
    du = zeros(2)
    # combined rhs is f1 + f2 = u + (-u) = 0
    prob.f(du, u0, nothing, 0.0)
    @test du ≈ zeros(2)

    # SplitODEProblem path still works and keeps a cache
    prob_split = SplitODEProblem(f1!, f2!, u0, (0.0, 1.0))
    @test prob_split.f._func_cache !== nothing
    du2 = zeros(2)
    prob_split.f(du2, u0, nothing, 0.0)
    @test du2 ≈ zeros(2)

    # out-of-place SplitFunction needs no cache
    f1 = (u, p, t) -> u
    f2 = (u, p, t) -> -u
    sf_oop = SplitFunction(f1, f2)
    @test sf_oop._func_cache === nothing
    prob_oop = ODEProblem(sf_oop, 1.0, (0.0, 1.0))
    @test prob_oop.f(1.0, nothing, 0.0) == 0.0
end

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

const ALIAS_SPECIFIER_TYPES = (
    SciMLBase.LinearAliasSpecifier,
    SciMLBase.NonlinearAliasSpecifier,
    SciMLBase.ODEAliasSpecifier,
    SciMLBase.RODEAliasSpecifier,
    SciMLBase.SDEAliasSpecifier,
    SciMLBase.DAEAliasSpecifier,
    SciMLBase.DDEAliasSpecifier,
    SciMLBase.SDDEAliasSpecifier,
    SciMLBase.BVPAliasSpecifier,
    SciMLBase.OptimizationAliasSpecifier,
    SciMLBase.IntegralAliasSpecifier,
    SciMLBase.DiscreteAliasSpecifier,
    SciMLBase.ImplicitDiscreteAliasSpecifier,
    SciMLBase.AnalyticalAliasSpecifier,
    SciMLBase.SteadyStateAliasSpecifier,
)

function alias_specifier_with_policy(T, alias)
    return if T === SciMLBase.IntegralAliasSpecifier
        T(nothing, nothing, alias)
    else
        T(; alias = alias)
    end
end

@testset "Alias specifier interface" begin
    @testset "$(nameof(T))" for T in ALIAS_SPECIFIER_TYPES
        @test T <: SciMLBase.AbstractAliasSpecifier

        default_spec = alias_specifier_with_policy(T, nothing)
        @test all(isnothing(getfield(default_spec, name)) for name in fieldnames(T))

        alias_spec = alias_specifier_with_policy(T, true)
        @test all(getfield(alias_spec, name) === true for name in fieldnames(T))

        noalias_spec = alias_specifier_with_policy(T, false)
        @test all(getfield(noalias_spec, name) === false for name in fieldnames(T))
    end
end

@testset "DAEProblem{iip, specialize} two-parameter constructor" begin
    function simple_dae!(resid, du, u, p, t)
        resid[1] = du[1] - u[1]
        resid[2] = u[1] + u[2] - 1.0
        return nothing
    end
    du0 = [1.0, 0.0]
    u0 = [1.0, 0.0]
    tspan = (0.0, 1.0)

    # Construction succeeds and locks in FullSpecialize
    prob = DAEProblem{true, SciMLBase.FullSpecialize}(simple_dae!, du0, u0, tspan)
    @test prob isa DAEProblem
    @test prob.f isa DAEFunction{true, SciMLBase.FullSpecialize}
    @test SciMLBase.specialization(prob.f) === SciMLBase.FullSpecialize
    @test SciMLBase.isinplace(prob) === true
    @test prob.du0 == du0
    @test prob.u0 == u0
    @test prob.tspan === (0.0, 1.0)

    # With explicit parameters and kwargs
    prob_p = DAEProblem{true, SciMLBase.FullSpecialize}(
        simple_dae!, du0, u0, tspan, 1.0; differential_vars = [true, false]
    )
    @test prob_p.p === 1.0
    @test prob_p.differential_vars == [true, false]
    @test SciMLBase.specialization(prob_p.f) === SciMLBase.FullSpecialize

    # f is callable through the wrapped DAEFunction
    resid = zeros(2)
    prob.f(resid, du0, u0, nothing, 0.0)
    @test resid[1] ≈ du0[1] - u0[1]
    @test resid[2] ≈ u0[1] + u0[2] - 1.0
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
        return SA[1 - y^2 - x^2]
    end

    function f2(u, p)
        yt = u[1]
        x, xt, y = p
        return SA[-2y * yt - 2x * xt]
    end

    function f3(u, p)
        lam = u[1]
        x, xt, y, yt = p
        return SA[-2xt^2 - 2yt^2 - 2y * (-1 + y * lam) - 2x^2 * lam]
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
        (prob1, prob2, prob3), (explicit1, explicit2, explicit3), p, true
    )

    @test !SciMLBase.isinplace(sccprob)
    @test sccprob isa SCCNonlinearProblem{SVector{3, Float64}}
    @test state_values(sccprob) isa SVector{3, Float64}
    @test sccprob.p === prob1.p === prob2.p === prob3.p

    sccprob2 = @inferred remake(sccprob; u0 = SA[2.0, 1.0, 2.0])
    @test !SciMLBase.isinplace(sccprob2)
    @test sccprob2 isa SCCNonlinearProblem{SVector{3, Float64}}
    @test state_values(sccprob2) isa SVector{3, Float64}
end

@testset "AutoDePSpecialize specialization marker" begin
    struct DePParams
        k::Float64
    end
    f_dep!(du, u, p, t) = (du[1] = -p.k * u[1]; nothing)

    prob = ODEProblem{true, SciMLBase.AutoDePSpecialize}(
        f_dep!, [1.0], (0.0, 1.0), DePParams(0.5)
    )
    @test prob isa ODEProblem
    @test prob.f isa ODEFunction{true, SciMLBase.AutoDePSpecialize}
    @test SciMLBase.specialization(prob.f) === SciMLBase.AutoDePSpecialize
    # The marker alone performs no wrapping or packing in SciMLBase: fields stay
    # concretely typed like FullSpecialize/AutoSpecialize construction, and p is
    # the user's struct until a solver path installs an opaque-parameter wrapper.
    @test prob.f.f === f_dep!
    @test prob.p === DePParams(0.5)
    du = [0.0]
    prob.f(du, [2.0], prob.p, 0.0)
    @test du[1] ≈ -1.0

    # Available for other function families, e.g. nonlinear problems.
    f_nl!(res, u, p) = (res[1] = u[1] - p.k; nothing)
    nlfun = NonlinearFunction{true, SciMLBase.AutoDePSpecialize}(f_nl!)
    @test SciMLBase.specialization(nlfun) === SciMLBase.AutoDePSpecialize

    # unwrapped_f reconstruction keeps the marker
    uf = SciMLBase.unwrapped_f(prob.f)
    @test SciMLBase.specialization(uf) === SciMLBase.AutoDePSpecialize
end
