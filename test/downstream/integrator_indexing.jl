using ModelingToolkit, OrdinaryDiffEq, RecursiveArrayTools, StochasticDiffEq,
      SymbolicIndexingInterface, Test
using ModelingToolkit: t_nounits as t, D_nounits as D
### Tests on non-layered model (everything should work). ###

@parameters a b c d
@variables s1(t) s2(t)

eqs = [D(s1) ~ a * s1 / (1 + s1 + s2) - b * s1,
    D(s2) ~ +c * s2 / (1 + s1 + s2) - d * s2]

@named population_model = ODESystem(eqs,t)

# Tests on ODEProblem.
u0 = [s1 => 2.0, s2 => 1.0]
p = [a => 2.0, b => 1.0, c => 1.0, d => 1.0]
tspan = (0.0, 1000000.0)
oprob = ODEProblem(population_model, u0, tspan, p)
integrator = init(oprob, Rodas4())

@test_throws Exception integrator[a]
@test_throws Exception integrator[population_model.a]
@test_throws Exception integrator[:a]
@test getp(oprob, a)(integrator) == getp(oprob, population_model.a)(integrator) ==
      getp(oprob, :a)(integrator) == 2.0
@test integrator.ps[a] == integrator.ps[population_model.a] == integrator.ps[:a] == 2.0
@test getp(oprob, b)(integrator) == getp(oprob, population_model.b)(integrator) ==
      getp(oprob, :b)(integrator) == 1.0
@test integrator.ps[b] == integrator.ps[population_model.b] == integrator.ps[:b] == 1.0
@test getp(oprob, c)(integrator) == getp(oprob, population_model.c)(integrator) ==
      getp(oprob, :c)(integrator) == 1.0
@test integrator.ps[d] == integrator.ps[population_model.d] == integrator.ps[:d] == 1.0
@test getp(oprob, d)(integrator) == getp(oprob, population_model.d)(integrator) ==
      getp(oprob, :d)(integrator) == 1.0
@test integrator.ps[d] == integrator.ps[population_model.d] == integrator.ps[:d] == 1.0

@test integrator[s1] == integrator[population_model.s1] == integrator[:s1] == 2.0
@test integrator[s2] == integrator[population_model.s2] == integrator[:s2] == 1.0
@test integrator[solvedvariables] == integrator.u
@test integrator[allvariables] == integrator.u
step!(integrator, 100.0, true)

@test getp(population_model, a)(integrator) ==
      getp(population_model, population_model.a)(integrator) ==
      getp(population_model, :a)(integrator) == 2.0
@test getp(population_model, b)(integrator) ==
      getp(population_model, population_model.b)(integrator) ==
      getp(population_model, :b)(integrator) == 1.0
@test getp(population_model, c)(integrator) ==
      getp(population_model, population_model.c)(integrator) ==
      getp(population_model, :c)(integrator) == 1.0
@test getp(population_model, d)(integrator) ==
      getp(population_model, population_model.d)(integrator) ==
      getp(population_model, :d)(integrator) == 1.0

@test integrator[s1] == integrator[population_model.s1] == integrator[:s1] != 2.0
@test integrator[s2] == integrator[population_model.s2] == integrator[:s2] != 1.0

setp(oprob, a)(integrator, 10.0)
@test getp(integrator, a)(integrator) == getp(integrator, population_model.a)(integrator) ==
      getp(integrator, :a)(integrator) == 10.0
@test integrator.ps[a] == integrator.ps[population_model.a] == integrator.ps[:a] == 10.0
setp(population_model, population_model.b)(integrator, 20.0)
@test getp(integrator, b)(integrator) == getp(integrator, population_model.b)(integrator) ==
      getp(integrator, :b)(integrator) == 20.0
@test integrator.ps[b] == integrator.ps[population_model.b] == integrator.ps[:b] == 20.0
setp(integrator, c)(integrator, 30.0)
@test getp(integrator, c)(integrator) == getp(integrator, population_model.c)(integrator) ==
      getp(integrator, :c)(integrator) == 30.0
@test integrator.ps[c] == integrator.ps[population_model.c] == integrator.ps[:c] == 30.0
integrator.ps[d] = 40.0
@test integrator.ps[d] == integrator.ps[population_model.d] == integrator.ps[:d] == 40.0

integrator[s1] = 10.0
@test integrator[s1] == integrator[population_model.s1] == integrator[:s1] == 10.0
integrator[population_model.s2] = 10.0
@test integrator[s2] == integrator[population_model.s2] == integrator[:s2] == 10.0
integrator[:s1] = 1.0
@test integrator[s1] == integrator[population_model.s1] == integrator[:s1] == 1.0

# Tests on SDEProblem
noiseeqs = [0.1 * s1,
    0.1 * s2]
@named noisy_population_model = SDESystem(population_model, noiseeqs)
sprob = SDEProblem(noisy_population_model, u0, (0.0, 100.0), p)
integrator = init(sprob, ImplicitEM())

step!(integrator, 100.0, true)

@test getp(sprob, a)(integrator) == getp(sprob, noisy_population_model.a)(integrator) ==
      getp(sprob, :a)(integrator) == 2.0
@test getp(sprob, b)(integrator) == getp(sprob, noisy_population_model.b)(integrator) ==
      getp(sprob, :b)(integrator) == 1.0
@test getp(sprob, c)(integrator) == getp(sprob, noisy_population_model.c)(integrator) ==
      getp(sprob, :c)(integrator) == 1.0
@test getp(sprob, d)(integrator) == getp(sprob, noisy_population_model.d)(integrator) ==
      getp(sprob, :d)(integrator) == 1.0
@test integrator[s1] == integrator[noisy_population_model.s1] == integrator[:s1] != 2.0
@test integrator[s2] == integrator[noisy_population_model.s2] == integrator[:s2] != 1.0

setp(integrator, a)(integrator, 10.0)
@test getp(noisy_population_model, a)(integrator) ==
      getp(noisy_population_model, noisy_population_model.a)(integrator) ==
      getp(noisy_population_model, :a)(integrator) == 10.0
setp(sprob, noisy_population_model.b)(integrator, 20.0)
@test getp(noisy_population_model, b)(integrator) ==
      getp(noisy_population_model, noisy_population_model.b)(integrator) ==
      getp(noisy_population_model, :b)(integrator) == 20.0
setp(noisy_population_model, c)(integrator, 30.0)
@test getp(noisy_population_model, c)(integrator) ==
      getp(noisy_population_model, noisy_population_model.c)(integrator) ==
      getp(noisy_population_model, :c)(integrator) == 30.0

integrator[s1] = 10.0
@test integrator[s1] == integrator[noisy_population_model.s1] == integrator[:s1] == 10.0
integrator[noisy_population_model.s2] = 10.0
@test integrator[s2] == integrator[noisy_population_model.s2] == integrator[:s2] == 10.0
integrator[:s1] = 1.0
@test integrator[s1] == integrator[noisy_population_model.s1] == integrator[:s1] == 1.0

@parameters t σ ρ β
@variables x(t) y(t) z(t)
D = Differential(t)

eqs = [D(x) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z]

@named lorenz1 = ODESystem(eqs,t)
@named lorenz2 = ODESystem(eqs,t)

@parameters γ
@variables a(t) α(t)
connections = [0 ~ lorenz1.x + lorenz2.y + a * γ,
    α ~ 2lorenz1.x + a * γ]
@mtkbuild sys_simplified = ODESystem(connections, t, [a, α], [γ], systems = [lorenz1, lorenz2])
sys_simplified = complete(structural_simplify(sys))

u0 = [lorenz1.x => 1.0,
    lorenz1.y => 0.0,
    lorenz1.z => 0.0,
    lorenz2.x => 0.0,
    lorenz2.y => 1.0,
    lorenz2.z => 0.0,
    a => 2.0]

p = [lorenz1.σ => 10.0,
    lorenz1.ρ => 28.0,
    lorenz1.β => 8 / 3,
    lorenz2.σ => 10.0,
    lorenz2.ρ => 28.0,
    lorenz2.β => 8 / 3,
    γ => 2.0]

tspan = (0.0, 100.0)
prob = ODEProblem(sys_simplified, u0, tspan, p)
integrator = init(prob, Rodas4())
step!(integrator, 100.0, true)

@test_throws Any integrator[b]
@test_throws Any integrator['a']

@test integrator[a] isa Real
@test_throws Any integrator[a, 1]
@test_throws Any integrator[a, 1:5]
@test_throws Any integrator[a, [1, 2, 3]]

@test integrator[1] isa Real
@test integrator[1:2] isa AbstractArray
@test integrator[[1, 2]] isa AbstractArray

@test integrator[lorenz1.x] isa Real
@test integrator[t] isa Real
@test integrator[α] isa Real
@test getp(prob, γ)(integrator) isa Real
@test getp(prob, γ)(integrator) == 2.0
@test getp(prob, (lorenz1.σ, lorenz1.ρ))(integrator) isa Tuple

@test length(integrator[[lorenz1.x, lorenz2.x]]) == 2
@test getp(integrator, [γ, lorenz1.σ])(integrator) isa Vector{Float64}
@test length(getp(integrator, [γ, lorenz1.σ])(integrator)) == 2

@variables q(t)[1:2] = [1.0, 2.0]
eqs = [D(q[1]) ~ 2q[1]
       D(q[2]) ~ 2.0]
@named sys2 = ODESystem(eqs, t, [q...], [])
sys2_simplified = complete(structural_simplify(sys2))
prob2 = ODEProblem(sys2, [], (0.0, 5.0))
integrator2 = init(prob2, Tsit5())

@test integrator2[q] isa Vector{Float64}
@test length(integrator2[q]) == length(q)
@test integrator2[collect(q)] == integrator2[q]
@test integrator2[(q...,)] isa NTuple{length(q), Float64}

@testset "Symbolic set_u!" begin
    @variables u(t)
    eqs = [D(u) ~ u]

    @mtkbuild sys2 = ODESystem(eqs,t)

    tspan = (0.0, 5.0)

    prob1 = ODEProblem(sys2, [u => 1.0], tspan)
    prob2 = ODEProblem(sys2, [u => 2.0], tspan)

    integrator1 = init(prob1, Tsit5(); save_everystep = false)
    integrator2 = init(prob2, Tsit5(); save_everystep = false)

    set_u!(integrator1, u, 2.0)

    @test integrator1.u ≈ integrator2.u
end

# Tests various interface methods:
@test_throws Any getp(sys, σ)(integrator)
@test in(getp(sys, lorenz1.σ)(integrator), integrator.p)
@test in(getp(sys, lorenz2.σ)(integrator), integrator.p)
@test_throws Any getp(sol, :σ)(sol)

@test_throws Any integrator[x]
@test in(integrator[lorenz1.x], integrator.u)
@test in(integrator[lorenz2.x], integrator.u)
@test_throws Any getp(sol, :x)(sol)

@test_throws Any setp(integrator, σ)(integrator, 2.0)
setp(integrator, lorenz1.σ)(integrator, 2.0)
@test getp(integrator, lorenz1.σ)(integrator) == 2.0
@test getp(integrator, lorenz2.σ)(integrator) != 2.0
setp(integrator, lorenz2.σ)(integrator, 2.0)
@test getp(integrator, lorenz2.σ)(integrator) == 2.0
@test_throws Any getp(sol, :σ)(sol)

@test_throws Any integrator[x]=2.0
integrator[lorenz1.x] = 2.0
@test integrator[lorenz1.x] == 2.0
@test integrator[lorenz2.x] != 2.0
integrator[lorenz2.x] = 2.0
@test integrator[lorenz2.x] == 2.0
@test_throws Any sol[:x]

# Check if indexing using variable names from interpolated integrator works
# It doesn't because this returns a Vector{Vector{T}} and not a DiffEqArray
# interpolated_integrator = integrator(0.0:1.0:10.0)
# @test interpolated_integrator[α] isa Vector
# @test interpolated_integrator[α, :] isa Vector
# @test interpolated_integrator[α, 2] isa Float64
# @test length(interpolated_integrator[α, 1:5]) == 5
# @test interpolated_integrator[α] ≈
#       2interpolated_integrator[lorenz1.x] .+ interpolated_integrator[a] .* 2.0
# @test collect(interpolated_integrator[t]) isa Vector
# @test collect(interpolated_integrator[t, :]) isa Vector
# @test interpolated_integrator[t, 2] isa Float64
# @test length(interpolated_integrator[t, 1:5]) == 5

# integrator1 = integrator(0.0:1.0:10.0)
# @test integrator1.u isa Vector
# @test first(integrator1.u) isa Vector
# @test length(integrator1.u) == 11
# @test length(integrator1.t) == 11

# integrator2 = integrator(0.1)
# @test integrator2 isa Vector
# @test length(integrator2) == length(unknowns(sys_simplified))
# @test first(integrator2) isa Real

# integrator3 = integrator(0.0:1.0:10.0, idxs = [lorenz1.x, lorenz2.x])
# @test integrator3.u isa Vector
# @test first(integrator3.u) isa Vector
# @test length(integrator3.u) == 11
# @test length(integrator3.t) == 11
# @test collect(integrator3[t]) ≈ integrator3.t
# @test collect(integrator3[t, 1:5]) ≈ integrator3.t[1:5]
# @test integrator(0.0:1.0:10.0, idxs = [lorenz1.x, 1]) isa RecursiveArrayTools.DiffEqArray

# integrator4 = integrator(0.1, idxs = [lorenz1.x, lorenz2.x])
# @test integrator4 isa Vector
# @test length(integrator4) == 2
# @test first(integrator4) isa Real
# @test integrator(0.1, idxs = [lorenz1.x, 1]) isa Vector{Real}

# integrator5 = integrator(0.0:1.0:10.0, idxs = lorenz1.x)
# @test integrator5.u isa Vector
# @test first(integrator5.u) isa Real
# @test length(integrator5.u) == 11
# @test length(integrator5.t) == 11
# @test collect(integrator5[t]) ≈ integrator3.t
# @test collect(integrator5[t, 1:5]) ≈ integrator3.t[1:5]
# @test_throws Any integrator(0.0:1.0:10.0, idxs = 1.2)

# integrator6 = integrator(0.1, idxs = lorenz1.x)
# @test integrator6 isa Real
# @test_throws Any integrator(0.1, idxs = 1.2)

# integrator7 = integrator(0.0:1.0:10.0, idxs = [2, 1])
# @test integrator7.u isa Vector
# @test first(integrator7.u) isa Vector
# @test length(integrator7.u) == 11
# @test length(integrator7.t) == 11
# @test collect(integrator7[t]) ≈ integrator3.t
# @test collect(integrator7[t, 1:5]) ≈ integrator3.t[1:5]

# integrator8 = integrator(0.1, idxs = [2, 1])
# @test integrator8 isa Vector
# @test length(integrator8) == 2
# @test first(integrator8) isa Real

# integrator9 = integrator(0.0:1.0:10.0, idxs = 2)
# @test integrator9.u isa Vector
# @test first(integrator9.u) isa Real
# @test length(integrator9.u) == 11
# @test length(integrator9.t) == 11
# @test collect(integrator9[t]) ≈ integrator3.t
# @test collect(integrator9[t, 1:5]) ≈ integrator3.t[1:5]

# integrator10 = integrator(0.1, idxs = 2)
# @test integrator10 isa Real

#=
using Plots
plot(sol,idxs=(lorenz2.x,lorenz2.z))
plot(sol,idxs=(α,lorenz2.z))
plot(sol,idxs=(lorenz2.x,α))
plot(sol,idxs=α)
plot(sol,idxs=(t,α))
=#

using LinearAlgebra
@variables t
sts = @variables x(t)[1:3]=[1, 2, 3.0] y(t)=1.0
ps = @parameters p[1:3] = [1, 2, 3]
D = Differential(t)
eqs = [collect(D.(x) .~ x)
       D(y) ~ norm(x) * y - x[1]]
@mtkbuild sys = ODESystem(eqs, t, [sts...;], [ps...;])
prob = ODEProblem(sys, [], (0, 1.0))
integrator = init(prob, Tsit5(), save_everystep = false)
@test integrator[x] isa Vector{Float64}
@test integrator[@nonamespace sys.x] isa Vector{Float64}

getx = getu(integrator, x)
gety = getu(integrator, :y)
get_arr = getu(integrator, [x, y])
get_tuple = getu(integrator, (x, y))
get_obs = getu(integrator, x[1] / p[1])
@test getx(integrator) == [1.0, 2.0, 3.0]
@test gety(integrator) == 1.0
@test get_arr(integrator) == [[1.0, 2.0, 3.0], 1.0]
@test get_tuple(integrator) == ([1.0, 2.0, 3.0], 1.0)
@test get_obs(integrator) == 1.0

setx! = setu(integrator, x)
sety! = setu(integrator, :y)
set_arr! = setu(integrator, [x, y])
set_tuple! = setu(integrator, (x, y))

setx!(integrator, [4.0, 5.0, 6.0])
@test getx(integrator) == [4.0, 5.0, 6.0]
sety!(integrator, 3.0)
@test gety(integrator) == 3.0
set_arr!(integrator, [[1.0, 2.0, 3.0], 1.0])
@test get_arr(integrator) == [[1.0, 2.0, 3.0], 1.0]
set_tuple!(integrator, ([2.0, 4.0, 6.0], 2.0))
@test get_tuple(integrator) == ([2.0, 4.0, 6.0], 2.0)
@test getp(sys, p)(integrator) == integrator.ps[p] == [1, 2, 3]
setp(sys, p)(integrator, [4, 5, 6])
@test getp(sys, p)(integrator) == integrator.ps[p] == [4, 5, 6]
integrator.ps[p] = [7, 8, 9]
@test getp(sys, p)(integrator) == integrator.ps[p] == [7, 8, 9]
