using ModelingToolkit, OrdinaryDiffEq, RecursiveArrayTools, Test

@parameters t σ ρ β
@variables x(t) y(t) z(t)
D = Differential(t)

eqs = [D(x) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z]

@named lorenz1 = ODESystem(eqs)
@named lorenz2 = ODESystem(eqs)

@parameters γ
@variables a(t) α(t)
connections = [0 ~ lorenz1.x + lorenz2.y + a * γ,
    α ~ 2lorenz1.x + a * γ]
@named sys = ODESystem(connections, t, [a, α], [γ], systems = [lorenz1, lorenz2])
sys_simplified = structural_simplify(sys)

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
@test integrator[γ] isa Real
@test integrator[γ] == 2.0
@test integrator[(lorenz1.σ, lorenz1.ρ)] isa Tuple

@test length(integrator[[lorenz1.x, lorenz2.x]]) == 2
@test integrator[[γ, lorenz1.σ]] isa Vector{Float64}
@test length(integrator[[γ, lorenz1.σ]]) == 2

@variables q(t)[1:2] = [1.0, 2.0]
eqs = [D(q[1]) ~ 2q[1]
       D(q[2]) ~ 2.0]
@named sys2 = ODESystem(eqs, t, [q...], [])
sys2_simplified = structural_simplify(sys2)
prob2 = ODEProblem(sys2, [], (0.0, 5.0))
integrator2 = init(prob2, Tsit5())

@test integrator2[q] isa Vector{Float64}
@test length(integrator2[q]) == length(q)
@test integrator2[collect(q)] == integrator2[q]
@test integrator2[(q...,)] isa NTuple{length(q), Float64}

@testset "Symbolic set_u!" begin
    @variables u(t)
    eqs = [D(u) ~ u]

    @named sys2 = ODESystem(eqs)

    tspan = (0.0, 5.0)

    prob1 = ODEProblem(sys2, [u => 1.0], tspan)
    prob2 = ODEProblem(sys2, [u => 2.0], tspan)

    integrator1 = init(prob1, Tsit5(); save_everystep = false)
    integrator2 = init(prob2, Tsit5(); save_everystep = false)

    set_u!(integrator1, u, 2.0)

    @test integrator1.u ≈ integrator2.u
end

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
# @test length(integrator2) == length(states(sys_simplified))
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
@named sys = ODESystem(eqs, t, [sts...;], [ps...;])
prob = ODEProblem(sys, [], (0, 1.0))
@test_broken local integrator = init(prob, Tsit5())
@test_broken integrator[x] isa Vector{<:Vector}
@test_broken integrator[@nonamespace sys.x] isa Vector{<:Vector}
