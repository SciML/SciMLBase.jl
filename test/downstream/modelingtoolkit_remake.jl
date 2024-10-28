using ModelingToolkit, SymbolicIndexingInterface
using JumpProcesses
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using Optimization
using OptimizationOptimJL

probs = []
syss = []

@parameters σ ρ β q
@variables x(t) y(t) z(t)

eqs = [D(x) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z]

@named sys = ODESystem(eqs, t; parameter_dependencies = [q => 3β])
sys = complete(sys)
u0 = [x => 1.0,
    y => 0.0,
    z => 0.0]

p = [σ => 28.0,
    ρ => 10.0,
    β => 8 / 3]

tspan = (0.0, 100.0)

push!(syss, sys)
push!(probs, ODEProblem(sys, u0, tspan, p, jac = true))

noise_eqs = [0.1x, 0.1y, 0.1z]
@named sdesys = SDESystem(sys, noise_eqs)
sdesys = complete(sdesys)

push!(syss, sdesys)
push!(probs, SDEProblem(sdesys, u0, tspan, p, jac = true))

@named nsys = NonlinearSystem([0 ~ eq.rhs for eq in eqs], [x, y, z], [σ, β, ρ])
nsys = complete(nsys)

push!(syss, nsys)
push!(probs, NonlinearProblem(nsys, u0, p, jac = true))

rate₁ = β * x * y
affect₁ = [x ~ x - σ, y ~ y + σ]
rate₂ = ρ * y
affect₂ = [y ~ y - 1, z ~ z + 1]
j₁ = ConstantRateJump(rate₁, affect₁)
j₂ = ConstantRateJump(rate₂, affect₂)
j₃ = MassActionJump(2 * β + ρ, [z => 1], [x => 1, z => -1])
@named js = JumpSystem([j₁, j₂, j₃], t, [x, y, z], [σ, β, ρ])
js = complete(js)
jump_dprob = DiscreteProblem(js, u0, tspan, p)

push!(syss, js)
push!(probs, JumpProblem(js, jump_dprob, Direct()))

@named optsys = OptimizationSystem(sum(eq.lhs for eq in eqs), [x, y, z], [σ, ρ, β])
optsys = complete(optsys)
push!(syss, optsys)
push!(probs, OptimizationProblem(optsys, u0, p))

k = ShiftIndex(t)
@mtkbuild discsys = DiscreteSystem(
    [x ~ x(k - 1) * ρ + y(k - 2), y ~ y(k - 1) * σ - z(k - 2), z ~ z(k - 1) * β + x(k - 2)],
    t)
# Roundabout method to avoid having to specify values for previous timestep
fn = DiscreteFunction(discsys)
ps = ModelingToolkit.MTKParameters(discsys, p)
discu0 = Dict([u0..., x(k - 1) => 0.0, y(k - 1) => 0.0, z(k - 1) => 0.0])
push!(syss, discsys)
push!(probs, DiscreteProblem(fn, getindex.((discu0,), unknowns(discsys)), (0, 10), ps))

for (sys, prob) in zip(syss, probs)
    @test parameter_values(prob) isa ModelingToolkit.MTKParameters

    @inferred typeof(prob) remake(prob)

    baseType = Base.typename(typeof(prob)).wrapper
    ugetter = getu(prob, [x, y, z])
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

# Optimization
@parameters p
@mtkbuild sys = ODESystem([D(x) ~ -p * x], t)
odeprob = ODEProblem(sys, [x => 1.0], (0.0, 10.0), [p => 0.5])

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
        @named sys = ODESystem([D(x) ~ P], t, [x], [P]; defaults = [P => sign * x]) # set P from initial condition of x
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
    @mtkbuild sys = ODESystem([D(x) ~ x, p ~ x + y], t)
    prob = ODEProblem(sys, [x => 1.0, y => 2.0], (0.0, 1.0))
    @test prob.ps[p] ≈ 3.0
    prob2 = remake(prob; u0 = [y => 3.0], p = Dict())
    @test prob2.ps[p] ≈ 4.0
end
