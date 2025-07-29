using ModelingToolkit, SymbolicIndexingInterface
using JumpProcesses
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using Optimization
using OptimizationOptimJL
using ForwardDiff
using SciMLStructures
using Test

probs = []
syss = []

@parameters σ ρ β q
@variables x(t) y(t) z(t)

eqs = [D(x) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z]

@named sys = System([eqs; q ~ 3β], t)
sys = complete(sys)
u0 = [x => 1.0,
    y => 0.0,
    z => 0.0]

p = [σ => 28.0,
    ρ => 10.0,
    β => 8 / 3]

tspan = (0.0, 100.0)

push!(syss, sys)
push!(probs, ODEProblem(sys, [u0; p], tspan, jac = true))

push!(syss, sys)
push!(probs, SteadyStateProblem(ODEProblem(sys, [u0; p], tspan, jac = true)))

noise_eqs = [0.1x, 0.1y, 0.1z]
@named sdesys = SDESystem(sys, noise_eqs)
sdesys = complete(sdesys)

push!(syss, sdesys)
push!(probs, SDEProblem(sdesys, [u0; p], tspan, jac = true))

@named nsys = System([0 ~ eq.rhs for eq in eqs], [x, y, z], [σ, β, ρ])
nsys = complete(nsys)

push!(syss, nsys)
push!(probs, NonlinearProblem(nsys, [u0; p], jac = true))

rate₁ = β * x * y
affect₁ = [x ~ x - σ, y ~ y + σ]
rate₂ = ρ * y
affect₂ = [y ~ y - 1, z ~ z + 1]
j₁ = ConstantRateJump(rate₁, affect₁)
j₂ = ConstantRateJump(rate₂, affect₂)
j₃ = MassActionJump(2 * β + ρ, [z => 1], [x => 1, z => -1])
@named js = JumpSystem([j₁, j₂, j₃], t, [x, y, z], [σ, β, ρ])
js = complete(js)

push!(syss, js)
push!(probs, JumpProblem(js, [u0; p], tspan; aggregator = Direct()))

@named optsys = OptimizationSystem(sum(eq.lhs for eq in eqs), [x, y, z], [σ, ρ, β])
optsys = complete(optsys)
push!(syss, optsys)
push!(probs, OptimizationProblem(optsys, [u0; p]))

@mtkcompile sys = System(
    [0 ~ x^3 * β + y^3 * ρ - σ, 0 ~ x^2 + 2x * y + y^2, 0 ~ z^2 - 4z + 4],
    [x, y, z], [σ, β, ρ])
sccprob = SCCNonlinearProblem(sys, [u0; p])
@test_nowarn SciMLBase.initialization_status(sccprob)
push!(syss, sys)
push!(probs, sccprob)

for (sys, prob) in zip(syss, probs)
    @test parameter_values(prob) isa ModelingToolkit.MTKParameters

    @inferred typeof(prob) remake(prob)

    baseType = Base.typename(typeof(prob)).wrapper
    ugetter = getsym(prob, [x, y, z])
    prob2 = @inferred baseType remake(prob; u0 = [x => 2.0, y => 3.0, z => 4.0])
    @test ugetter(prob2) == [2.0, 3.0, 4.0]
    prob2 = @inferred baseType remake(prob; u0 = [sys.x => 2.0, sys.y => 3.0, sys.z => 4.0])
    @test ugetter(prob2) == [2.0, 3.0, 4.0]
    prob2 = @inferred baseType remake(prob; u0 = [:x => 2.0, :y => 3.0, :z => 4.0])
    @test ugetter(prob2) == [2.0, 3.0, 4.0]
    prob2 = @inferred baseType remake(prob; u0 = [x => 2.0, sys.y => 3.0, :z => 4.0])
    @test ugetter(prob2) == [2.0, 3.0, 4.0]

    prob2 = @inferred baseType remake(prob; u0 = [x => 12.0])
    @test ugetter(prob2) == [12.0, 0.0, 0.0]
    prob2 = @inferred baseType remake(prob; u0 = [sys.x => 12.0])
    @test ugetter(prob2) == [12.0, 0.0, 0.0]
    prob2 = @inferred baseType remake(prob; u0 = [:x => 12.0])
    @test ugetter(prob2) == [12.0, 0.0, 0.0]

    pgetter = getp(prob, [σ, β, ρ])
    prob2 = @inferred baseType remake(prob; p = [σ => 0.1, β => 0.2, ρ => 0.3])
    @test pgetter(prob2) == [0.1, 0.2, 0.3]
    if prob isa ODEProblem
        @test prob2.ps[q] ≈ 0.6
    end
    prob2 = @inferred baseType remake(prob; p = [sys.σ => 0.1, sys.β => 0.2, sys.ρ => 0.3])
    @test pgetter(prob2) == [0.1, 0.2, 0.3]
    if prob isa ODEProblem
        @test prob2.ps[q] ≈ 0.6
    end
    prob2 = @inferred baseType remake(prob; p = [:σ => 0.1, :β => 0.2, :ρ => 0.3])
    @test pgetter(prob2) == [0.1, 0.2, 0.3]
    if prob isa ODEProblem
        @test prob2.ps[q] ≈ 0.6
    end
    prob2 = @inferred baseType remake(prob; p = [σ => 0.1, sys.β => 0.2, :ρ => 0.3])
    @test pgetter(prob2) == [0.1, 0.2, 0.3]
    if prob isa ODEProblem
        @test prob2.ps[q] ≈ 0.6
    end

    prob2 = @inferred baseType remake(prob; p = [σ => 0.5])
    @test pgetter(prob2) == [0.5, 8 / 3, 10.0]
    prob2 = @inferred baseType remake(prob; p = [sys.σ => 0.5])
    @test pgetter(prob2) == [0.5, 8 / 3, 10.0]
    prob2 = @inferred baseType remake(prob; p = [:σ => 0.5])
    @test pgetter(prob2) == [0.5, 8 / 3, 10.0]

    # Test p dependent on u0
    prob2 = @inferred baseType remake(prob; p = [σ => 0.5x + 1])
    @test pgetter(prob2) ≈ [1.5, 8 / 3, 10.0]
    prob2 = @inferred baseType remake(prob; p = [sys.σ => 0.5x + 1])
    @test pgetter(prob2) ≈ [1.5, 8 / 3, 10.0]
    prob2 = @inferred baseType remake(prob; p = [:σ => 0.5x + 1])
    @test pgetter(prob2) ≈ [1.5, 8 / 3, 10.0]

    # Test u0 dependent on p
    prob2 = @inferred baseType remake(prob; u0 = [x => 0.5σ + 1])
    @test ugetter(prob2) ≈ [15.0, 0.0, 0.0]
    prob2 = @inferred baseType remake(prob; u0 = [sys.x => 0.5σ + 1])
    @test ugetter(prob2) ≈ [15.0, 0.0, 0.0]
    prob2 = @inferred baseType remake(prob; u0 = [:x => 0.5σ + 1])
    @test ugetter(prob2) ≈ [15.0, 0.0, 0.0]

    # Test u0 dependent on p and p dependent on u0
    prob2 = @inferred baseType remake(prob; u0 = [x => 0.5σ + 1], p = [β => 0.5x + 1])
    @test ugetter(prob2) ≈ [15.0, 0.0, 0.0]
    @test pgetter(prob2) ≈ [28.0, 8.5, 10.0]
    prob2 = @inferred baseType remake(
        prob; u0 = [sys.x => 0.5σ + 1], p = [sys.β => 0.5x + 1])
    @test ugetter(prob2) ≈ [15.0, 0.0, 0.0]
    @test pgetter(prob2) ≈ [28.0, 8.5, 10.0]
    # Not testing `Symbol => expr` since nested substitution doesn't work with that
end

@testset "DiscreteProblem" begin
    k = ShiftIndex(t)
    @mtkcompile discsys = System(
        [x ~ x(k - 1) * ρ + y(k - 2), y ~ y(k - 1) * σ - z(k - 2),
            z ~ z(k - 1) * β + x(k - 2)],
        t; defaults = [
            x => 1.0, y => 1.0, z => 1.0, x(k-1) => 0.0, y(k-1) => 0.0, z(k-1) => 0.0])
    prob = DiscreteProblem(discsys, p, (0, 10))
    prob[x(k-1)] = 1.0
    prob[y(k-1)] = prob[z(k-1)] = 0.0

    @test parameter_values(prob) isa ModelingToolkit.MTKParameters
    @inferred typeof(prob) remake(prob)

    baseType = Base.typename(typeof(prob)).wrapper
    ugetter = getsym(prob, [x(k-1), y(k-1), z(k-1)])
    prob2 = @inferred baseType remake(
        prob; u0 = [x(k-1) => 2.0, y(k-1) => 3.0, z(k-1) => 4.0])
    @test ugetter(prob2) == [2.0, 3.0, 4.0]
    prob2 = @inferred baseType remake(
        prob; u0 = [sys.x(k-1) => 2.0, sys.y(k-1) => 3.0, sys.z(k-1) => 4.0])
    @test ugetter(prob2) == [2.0, 3.0, 4.0]
    prob2 = @inferred baseType remake(prob; u0 = [:xₜ₋₁ => 2.0, :yₜ₋₁ => 3.0, :zₜ₋₁ => 4.0])
    @test ugetter(prob2) == [2.0, 3.0, 4.0]
    prob2 = @inferred baseType remake(
        prob; u0 = [x(k-1) => 2.0, sys.y(k-1) => 3.0, :zₜ₋₁ => 4.0])
    @test ugetter(prob2) == [2.0, 3.0, 4.0]

    prob2 = @inferred baseType remake(prob; u0 = [x(k-1) => 12.0])
    @test ugetter(prob2) == [12.0, 0.0, 0.0]
    prob2 = @inferred baseType remake(prob; u0 = [sys.x(k-1) => 12.0])
    @test ugetter(prob2) == [12.0, 0.0, 0.0]
    prob2 = @inferred baseType remake(prob; u0 = [:xₜ₋₁ => 12.0])
    @test ugetter(prob2) == [12.0, 0.0, 0.0]

    pgetter = getp(prob, [σ, β, ρ])
    prob2 = @inferred baseType remake(prob; p = [σ => 0.1, β => 0.2, ρ => 0.3])
    @test pgetter(prob2) == [0.1, 0.2, 0.3]
    if prob isa ODEProblem
        @test prob2.ps[q] ≈ 0.6
    end
    prob2 = @inferred baseType remake(prob; p = [sys.σ => 0.1, sys.β => 0.2, sys.ρ => 0.3])
    @test pgetter(prob2) == [0.1, 0.2, 0.3]
    if prob isa ODEProblem
        @test prob2.ps[q] ≈ 0.6
    end
    prob2 = @inferred baseType remake(prob; p = [:σ => 0.1, :β => 0.2, :ρ => 0.3])
    @test pgetter(prob2) == [0.1, 0.2, 0.3]
    if prob isa ODEProblem
        @test prob2.ps[q] ≈ 0.6
    end
    prob2 = @inferred baseType remake(prob; p = [σ => 0.1, sys.β => 0.2, :ρ => 0.3])
    @test pgetter(prob2) == [0.1, 0.2, 0.3]
    if prob isa ODEProblem
        @test prob2.ps[q] ≈ 0.6
    end

    prob2 = @inferred baseType remake(prob; p = [σ => 0.5])
    @test pgetter(prob2) == [0.5, 8 / 3, 10.0]
    prob2 = @inferred baseType remake(prob; p = [sys.σ => 0.5])
    @test pgetter(prob2) == [0.5, 8 / 3, 10.0]
    prob2 = @inferred baseType remake(prob; p = [:σ => 0.5])
    @test pgetter(prob2) == [0.5, 8 / 3, 10.0]

    # Test p dependent on u0
    @test_broken begin
        prob2 = @inferred baseType remake(prob; p = [σ => 0.5x(k-1) + 1])
        @test pgetter(prob2) ≈ [1.5, 8 / 3, 10.0]
        prob2 = @inferred baseType remake(prob; p = [sys.σ => 0.5x(k-1) + 1])
        @test pgetter(prob2) ≈ [1.5, 8 / 3, 10.0]
        prob2 = @inferred baseType remake(prob; p = [:σ => 0.5x(k-1) + 1])
        @test pgetter(prob2) ≈ [1.5, 8 / 3, 10.0]
    end

    # Test u0 dependent on p
    prob2 = @inferred baseType remake(prob; u0 = [x(k-1) => 0.5σ + 1])
    @test ugetter(prob2) ≈ [15.0, 0.0, 0.0]
    prob2 = @inferred baseType remake(prob; u0 = [sys.x(k-1) => 0.5σ + 1])
    @test ugetter(prob2) ≈ [15.0, 0.0, 0.0]
    prob2 = @inferred baseType remake(prob; u0 = [:xₜ₋₁ => 0.5σ + 1])
    @test ugetter(prob2) ≈ [15.0, 0.0, 0.0]

    # Test u0 dependent on p and p dependent on u0
    prob2 = @inferred baseType remake(
        prob; u0 = [x(k-1) => 0.5σ + 1], p = [β => 0.5x(k-1) + 1])
    @test ugetter(prob2) ≈ [15.0, 0.0, 0.0]
    @test_broken pgetter(prob2) ≈ [28.0, 8.5, 10.0]
    prob2 = @inferred baseType remake(
        prob; u0 = [sys.x(k-1) => 0.5σ + 1], p = [sys.β => 0.5x(k-1) + 1])
    @test ugetter(prob2) ≈ [15.0, 0.0, 0.0]
    @test_broken pgetter(prob2) ≈ [28.0, 8.5, 10.0]
    # Not testing `Symbol => expr` since nested substitution doesn't work with that
end

# Optimization
@parameters p
@mtkcompile sys = System([D(x) ~ -p * x], t)
odeprob = ODEProblem(sys, [x => 1.0, p => 0.5], (0.0, 10.0))

ts = 0.0:0.5:10.0
data = exp.(-2.5 .* ts)

function loss(x, p)
    prob = p[1]

    prob = @inferred ODEProblem remake(
        prob; p = [prob.f.sys.p => x[1]], u0 = typeof(x)(prob.u0))
    sol = solve(prob, Tsit5())
    vals = sol(ts; idxs = prob.f.sys.x).u
    return sum((data .- vals) .^ 2) / length(ts)
end

f = OptimizationFunction(loss, Optimization.AutoForwardDiff())
prob = OptimizationProblem(f, [0.5], [odeprob])
sol = solve(prob, BFGS())
@test sol.u[1]≈2.5 rtol=1e-4

# Issue ModelingToolkit.jl#2637
@testset "remake with defaults containing expressions" begin
    @variables x(t)
    @parameters P

    for sign in [+1.0, -1.0]
        @named sys = System([D(x) ~ P], t, [x], [P]; defaults = [P => sign * x]) # set P from initial condition of x
        sys = complete(sys)

        prob1 = ODEProblem(sys, [x => 1.0], (0.0, 1.0))
        @test prob1.ps[P] == sign * 1.0

        prob2 = remake(prob1; u0 = [x => 2.0], p = Dict(), use_defaults = true) # use defaults to update parameters
        @test prob2.ps[P] == sign * 2.0
    end
end

@testset "remake with Vector{Int} as index of array variable/parameter" begin
    @parameters k[1:4]
    @variables (V(t))[1:2]
    function rhs!(du, u, p, t)
        du[1] = p[1] - p[2] * u[1]
        du[2] = p[3] - p[4] * u[2]
        nothing
    end
    sys = SymbolCache(Dict(V => 1:2, V[1] => 1, V[2] => 2),
        Dict(k => 1:4, k[1] => 1, k[2] => 2, k[3] => 3, k[4] => 4), t)
    struct SCWrapper{S}
        sys::S
    end
    SymbolicIndexingInterface.symbolic_container(s::SCWrapper) = s.sys
    SymbolicIndexingInterface.variable_symbols(s::SCWrapper) = filter(
        x -> symbolic_type(x) != ArraySymbolic(), variable_symbols(s.sys))
    SymbolicIndexingInterface.parameter_symbols(s::SCWrapper) = filter(
        x -> symbolic_type(x) != ArraySymbolic(), parameter_symbols(s.sys))
    sys = SCWrapper(sys)
    fn = ODEFunction(rhs!; sys)
    oprob_scal_scal = ODEProblem(fn, [10.0, 20.0], (0.0, 1.0), [1.0, 2.0, 3.0, 4.0])
    ps_vec = [k => [2.0, 3.0, 4.0, 5.0]]
    u0_vec = [V => [1.5, 2.5]]
    newoprob = remake(oprob_scal_scal; u0 = u0_vec, p = ps_vec)
    @test newoprob.ps[k] == [2.0, 3.0, 4.0, 5.0]
    @test newoprob[V] == [1.5, 2.5]
end

@testset "remake with parameter dependent on observed" begin
    @variables x(t) y(t)
    @parameters p = x + y
    @mtkcompile sys = System([D(x) ~ x, p ~ x + y], t)
    prob = ODEProblem(sys, [x => 1.0, y => 2.0], (0.0, 1.0))
    @test prob.ps[p] ≈ 3.0
    prob2 = remake(prob; u0 = [y => 3.0], p = Dict())
    @test prob2.ps[p] ≈ 4.0
end

@testset "u0 dependent on parameter given as Symbol" begin
    @variables x(t)
    @parameters p
    @mtkcompile sys = System([D(x) ~ x * p], t)
    prob = ODEProblem(sys, [x => 1.0, p => 1.0], (0.0, 1.0))
    @test prob.ps[p] ≈ 1.0
    prob2 = remake(prob; u0 = [x => p], p = [:p => 2.0])
    @test prob2[x] ≈ 2.0
end

@testset "remake dependent on indepvar" begin
    @variables x(t)
    @parameters p
    @mtkcompile sys = System([D(x) ~ x * p], t)
    prob = ODEProblem(sys, [x => 1.0, p => 1.0], (0.0, 1.0))
    prob2 = remake(prob; u0 = [x => t + 3.0])
    @test prob2[x] ≈ 3.0
end

@static if length(methods(SciMLBase.detect_cycles)) == 1
    function SciMLBase.detect_cycles(
            ::ModelingToolkit.AbstractSystem, varmap::Dict{Any, Any}, vars)
        for sym in vars
            newval = ModelingToolkit.fixpoint_sub(sym, varmap; maxiters = 10)
            vs = ModelingToolkit.vars(newval)
            if !isempty(vars) && any(in(Set(vars)), vs)
                NotSymbolic()
                return true
            end
        end
        return false
    end
end

@testset "Cycle detection" begin
    @variables x(t) y(t)
    @parameters p q
    @mtkcompile sys = System([D(x) ~ x * p, D(y) ~ y * q], t)

    prob = ODEProblem(sys, [x => 1.0, y => 1.0, p => 1.0, q => 1.0], (0.0, 1.0))
    @test_throws SciMLBase.CyclicDependencyError remake(
        prob; u0 = [x => 2y + 3, y => 2x + 1])
    @test_throws SciMLBase.CyclicDependencyError remake(prob; p = [p => 2q + 1, q => p + 3])
    @test_throws SciMLBase.CyclicDependencyError remake(
        prob; u0 = [x => 2y + p, y => q + 3], p = [p => x + y, q => p + 3])
end

@testset "SCCNonlinearProblem" begin
    @mtkbuild fullsys = System(
        [0 ~ x^3 * β + y^3 * ρ - σ, 0 ~ x^2 + 2x * y + y^2, 0 ~ z^2 - 4z + 4],
        [x, y, z], [σ, β, ρ])

    u0 = [x => 1.0,
        y => 0.0,
        z => 0.0]

    p = [σ => 28.0,
        ρ => 10.0,
        β => 8 / 3]

    sccprob = SCCNonlinearProblem(fullsys, [u0; p])

    sccprob2 = remake(sccprob; u0 = 2ones(3))
    @test state_values(sccprob2) ≈ 2ones(3)
    prob1, prob2 = if length(state_values(sccprob2.probs[1])) == 1
        sccprob2.probs[2], sccprob2.probs[1]
    else
        sccprob2.probs[1], sccprob2.probs[2]
    end
    @test prob1.u0 ≈ 2ones(2)
    @test prob2.u0 ≈ 2ones(1)
    @test sccprob2.explicitfuns! !== missing
    @test sccprob2.f.sys !== missing

    sccprob3 = remake(sccprob; p = [σ => 2.0])
    @test sccprob3.p === sccprob3.probs[1].p
    @test sccprob3.p === sccprob3.probs[2].p

    @test_throws ["parameters_alias", "SCCNonlinearProblem"] remake(
        sccprob; parameters_alias = false, p = [σ => 2.0])

    newp = remake_buffer(sccprob.f.sys, sccprob.p, [σ], [3.0])
    sccprob4 = remake(sccprob; parameters_alias = false, p = newp,
        probs = [remake(sccprob.probs[1]; p = deepcopy(newp)), sccprob.probs[2]])
    @test sccprob4.parameters_alias === Val(false)
    @test sccprob4.p !== sccprob4.probs[1].p
    @test sccprob4.p !== sccprob4.probs[2].p
end

# TODO: Rewrite this test when MTK build initialization for everything
@testset "Lazy initialization" begin
    @variables _x(..) [guess = 1.0] y(t) [guess = 1.0]
    @parameters p=1.0 [guess = 1.0]
    @brownians a
    x = _x(t)

    initprob = NonlinearProblem(nothing) do args...
        return 0.0
    end
    iprobmap = (_...) -> [1.0, 1.0]
    iprobpmap = function (orig, sol)
        ps = parameter_values(orig)
        setp(orig, p)(ps, 3.0)
        return ps
    end
    initdata = SciMLBase.OverrideInitData(initprob, nothing, iprobmap, iprobpmap)
    @test SciMLBase.is_trivial_initialization(initdata)

    @testset "$Problem" for (rhss, Problem, Func) in [
        (0.0, ODEProblem, ODEFunction),
        (a, SDEProblem, SDEFunction),
        (_x(t - 0.1), DDEProblem, DDEFunction),
        (_x(t - 0.1) + a, SDDEProblem, SDDEFunction),
        (y + 2, NonlinearProblem, NonlinearFunction),
        (y + 2, NonlinearLeastSquaresProblem, NonlinearFunction)
    ]
        is_nlsolve = Func == NonlinearFunction
        D = is_nlsolve ? (v) -> v^3 : Differential(t)
        sys_args = is_nlsolve ? () : (t,)
        prob_args = is_nlsolve ? () : ((0.0, 1.0),)

        @mtkcompile sys = System([D(x) ~ x + rhss, x + y ~ p], sys_args...)
        prob = Problem(sys, [x => 1.0, y => 1.0], prob_args...)
        func_args = isdefined(prob.f, :g) ? (prob.f.g,) : ()
        func = Func{true, SciMLBase.FullSpecialize}(
            prob.f.f, func_args...; initialization_data = initdata, sys = prob.f.sys)
        prob2 = remake(prob; f = func)
        @test SciMLBase.is_trivial_initialization(prob2)
        @test prob2.ps[p] ≈ 3.0
    end
end

# https://github.com/SciML/ModelingToolkit.jl/issues/3326
@testset "Remake with non-substitute-able values" begin
    @variables x1(t) x2(t) y(t)
    @parameters k2 p Gamma y0 d k1
    @mtkcompile sys = System(
        [D(y) ~ p - d * y, D(x1) ~ -k1 * x1 + k2 * (Gamma - x1), x2 ~ Gamma - x1],
        t; defaults = Dict(y => y0, Gamma => x1 + x2))
    u0 = [x1 => 1.0, x2 => 2.0]
    p0 = [p => 10.0, d => 5.0, y0 => 3.0, k1 => 1.0, k2 => 2.0]
    prob = ODEProblem(sys, [u0; p0], (0.0, 1.0))
    u0_new = [x1 => 0.1]
    ps_new = [y0 => 0.3, p => 100.0]
    prob2 = remake(prob; u0 = u0_new, p = ps_new)
    @test prob2[x1] ≈ 0.1
    @test prob2[y] ≈ 0.3
    # old value retained
    @test prob2.ps[Gamma] ≈ 3.0
end

@testset "`nothing` value for variable specified as `Symbol`" begin
    @variables x(t) [guess = 1.0] y(t) [guess = 1.0]
    @parameters p [guess = 1.0] q [guess = 1.0]
    @mtkcompile sys = System(
        [D(x) ~ p * x + q * y, y ~ 2x], t; parameter_dependencies = [q ~ 2p])
    prob = ODEProblem(sys, [:x => 1.0, p => 1.0], (0.0, 1.0))
    @test_nowarn remake(prob; u0 = [:y => 1.0, :x => nothing])
end

@testset "`initialization_data` u0 and p are promoted with explicit `f`" begin
    @variables x(t) [guess = 1.0] y(t) [guess = 1.0]
    @parameters p q
    @mtkcompile sys = System([D(x) ~ x, (x - p)^2 + (y - q)^3 ~ 0], t)
    prob = ODEProblem(sys, [x => 1.0, p => 1.0, q => 2.0], (0.0, 1.0))
    @test prob.f.initialization_data !== nothing
    buf, repack, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)
    newps = repack(ForwardDiff.Dual.(buf))
    prob2 = @test_nowarn remake(
        prob; f = prob.f, u0 = ForwardDiff.Dual.(prob.u0), p = newps)
    @test prob2.f.initialization_data !== nothing
    initprob = prob2.f.initialization_data.initializeprob
    @test eltype(initprob.u0) <: ForwardDiff.Dual
    @test eltype(SciMLStructures.canonicalize(SciMLStructures.Tunable(), initprob.p)[1]) <:
          ForwardDiff.Dual
end

@testset "Array unknown specified as Symbol" begin
    @variables x(t)[1:2]
    @parameters k
    @mtkcompile sys = System([D(x[1]) ~ k * x[1], D(x[2]) ~ -x[2]], t)
    prob = ODEProblem(sys, [x => ones(2), k => 2.0], (0.0, 1.0))
    prob2 = remake(prob; u0 = [:x => 2ones(2)])
end
