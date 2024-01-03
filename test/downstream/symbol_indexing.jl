using ModelingToolkit, OrdinaryDiffEq, RecursiveArrayTools, SymbolicIndexingInterface, Test
using Optimization, OptimizationOptimJL

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
sol = solve(prob, Rodas4())

@test_throws Any sol[b]
@test_throws Any sol[b, 1]
@test_throws Any sol[b, 1:5]
@test_throws Any sol[b, [1, 2, 3]]
@test_throws Any sol['a']
@test_throws Any sol['a', 1]
@test_throws Any sol['a', 1:5]
@test_throws Any sol['a', [1, 2, 3]]

@test sol[a] isa AbstractVector
@test sol[:a] == sol[a]
@test sol[a, 1] isa Real
@test sol[:a, 1] == sol[a, 1]
@test sol[a, 1:5] isa AbstractVector
@test sol[:a, 1:5] == sol[a, 1:5]
@test sol[a, [1, 2, 3]] isa AbstractVector
@test sol[:a, [1, 2, 3]] == sol[a, [1, 2, 3]]

@test sol[:, 1] isa AbstractVector
@test sol[:, 1:2] isa AbstractDiffEqArray
@test sol[:, [1, 2]] isa AbstractDiffEqArray

@test sol[lorenz1.x] isa Vector
@test sol[lorenz1.x, 2] isa Float64
@test sol[lorenz1.x, :] isa Vector
@test sol[t] isa Vector
@test sol[t, 2] isa Float64
@test sol[t, :] isa Vector
@test length(sol[lorenz1.x, 1:5]) == 5
@test sol[α] isa Vector
@test sol[α, 3] isa Float64
@test length(sol[α, 5:10]) == 6
@test getp(prob, γ)(sol) isa Real
@test sol.ps[γ] isa Real
@test getp(prob, γ)(sol) == getp(prob, :γ)(sol) == sol.ps[γ] == sol.ps[:γ] == 2.0
@test getp(prob, (lorenz1.σ, lorenz1.ρ))(sol) isa Tuple
@test sol.ps[(lorenz1.σ, lorenz1.ρ)] isa Tuple

@test sol[[lorenz1.x, lorenz2.x]] isa Vector{Vector{Float64}}
@test length(sol[[lorenz1.x, lorenz2.x]]) == length(sol)
@test all(length.(sol[[lorenz1.x, lorenz2.x]]) .== 2)
@test sol[(lorenz1.x, lorenz2.x)] isa Vector{Tuple{Float64, Float64}}
@test length(sol[(lorenz1.x, lorenz2.x)]) == length(sol)
@test all(length.(sol[(lorenz1.x, lorenz2.x)]) .== 2)

@test sol[[lorenz1.x, lorenz2.x], :] isa Matrix{Float64}
@test size(sol[[lorenz1.x, lorenz2.x], :]) == (2, length(sol))
@test size(sol[[lorenz1.x, lorenz2.x], :]) == size(sol[[1, 2], :]) == size(sol[1:2, :])

@variables q(t)[1:2] = [1.0, 2.0]
eqs = [D(q[1]) ~ 2q[1]
    D(q[2]) ~ 2.0]
@named sys2 = ODESystem(eqs, t, [q...], [])
sys2_simplified = structural_simplify(sys2)
prob2 = ODEProblem(sys2, [], (0.0, 5.0))
sol2 = solve(prob2, Tsit5())

@test sol2[q] isa Vector{Vector{Float64}}
@test sol2[(q...,)] isa Vector{NTuple{length(q), Float64}}
@test length(sol2[q]) == length(sol2)
@test all(length.(sol2[q]) .== 2)
@test sol2[collect(q)] == sol2[q]

# Check if indexing using variable names from interpolated solution works
interpolated_sol = sol(0.0:1.0:10.0)
@test interpolated_sol[α] isa Vector
@test interpolated_sol[α, :] isa Vector
@test interpolated_sol[α, 2] isa Float64
@test length(interpolated_sol[α, 1:5]) == 5
@test interpolated_sol[α] ≈ 2interpolated_sol[lorenz1.x] .+ interpolated_sol[a] .* 2.0
@test collect(interpolated_sol[t]) isa Vector
@test collect(interpolated_sol[t, :]) isa Vector
@test interpolated_sol[t, 2] isa Float64
@test length(interpolated_sol[t, 1:5]) == 5

sol1 = sol(0.0:1.0:10.0)
@test sol1.u isa Vector
@test first(sol1.u) isa Vector
@test length(sol1.u) == 11
@test length(sol1.t) == 11

sol2 = sol(0.1)
@test sol2 isa Vector
@test length(sol2) == length(states(sys_simplified))
@test first(sol2) isa Real

sol3 = sol(0.0:1.0:10.0, idxs = [lorenz1.x, lorenz2.x])
@test sol3.u isa Vector
@test first(sol3.u) isa Vector
@test length(sol3.u) == 11
@test length(sol3.t) == 11
@test collect(sol3[t]) ≈ sol3.t
@test collect(sol3[t, 1:5]) ≈ sol3.t[1:5]
@test sol(0.0:1.0:10.0, idxs = [lorenz1.x, 1]) isa RecursiveArrayTools.DiffEqArray

sol4 = sol(0.1, idxs = [lorenz1.x, lorenz2.x])
@test sol4 isa Vector
@test length(sol4) == 2
@test first(sol4) isa Real
@test sol(0.1, idxs = [lorenz1.x, 1]) isa Vector{<:Real}

sol5 = sol(0.0:1.0:10.0, idxs = lorenz1.x)
@test sol5.u isa Vector
@test first(sol5.u) isa Real
@test length(sol5.u) == 11
@test length(sol5.t) == 11
@test collect(sol5[t]) ≈ sol3.t
@test collect(sol5[t, 1:5]) ≈ sol3.t[1:5]
@test_throws Any sol(0.0:1.0:10.0, idxs = 1.2)

sol6 = sol(0.1, idxs = lorenz1.x)
@test sol6 isa Real
@test_throws Any sol(0.1, idxs = 1.2)

sol7 = sol(0.0:1.0:10.0, idxs = [2, 1])
@test sol7.u isa Vector
@test first(sol7.u) isa Vector
@test length(sol7.u) == 11
@test length(sol7.t) == 11
@test collect(sol7[t]) ≈ sol3.t
@test collect(sol7[t, 1:5]) ≈ sol3.t[1:5]

sol8 = sol(0.1, idxs = [2, 1])
@test sol8 isa Vector
@test length(sol8) == 2
@test first(sol8) isa Real

sol9 = sol(0.0:1.0:10.0, idxs = 2)
@test sol9.u isa Vector
@test first(sol9.u) isa Real
@test length(sol9.u) == 11
@test length(sol9.t) == 11
@test collect(sol9[t]) ≈ sol3.t
@test collect(sol9[t, 1:5]) ≈ sol3.t[1:5]

sol10 = sol(0.1, idxs = 2)
@test sol10 isa Real

@test is_timeseries(sol) == Timeseries()
getx = getu(sys_simplified, lorenz1.x)
get_arr = getu(sys_simplified, [lorenz1.x, lorenz2.x])
get_tuple = getu(sys_simplified, (lorenz1.x, lorenz2.x))
get_obs = getu(sol, lorenz1.x + lorenz2.x) # can't use sys for observed
get_obs_arr = getu(sol, [lorenz1.x + lorenz2.x, lorenz1.y + lorenz2.y])
l1x_idx = variable_index(sol, lorenz1.x)
l2x_idx = variable_index(sol, lorenz2.x)
l1y_idx = variable_index(sol, lorenz1.y)
l2y_idx = variable_index(sol, lorenz2.y)

@test getx(sol) == sol[:, l1x_idx]
@test get_arr(sol) == sol[:, [l1x_idx, l2x_idx]]
@test get_tuple(sol) == tuple.(sol[:, l1x_idx], sol[:, l2x_idx])
@test get_obs(sol) == sol[:, l1x_idx] + sol[:, l2x_idx]
@test get_obs_arr(sol) == vcat.(sol[:, l1x_idx] + sol[:, l2x_idx], sol[:, l1y_idx] + sol[:, l2y_idx])

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
sol = solve(prob, Tsit5())
@test sol[x] isa Vector{<:Vector}
@test sol[@nonamespace sys.x] isa Vector{<:Vector}
@test sol.ps[p] == [1, 2, 3]

getx = getu(sys, x)
get_mix_arr = getu(sys, [x, y])
get_mix_tuple = getu(sys, (x, y))
x_idx = variable_index.((sys,), [x[1], x[2], x[3]])
y_idx = variable_index(sys, y)
@test getx(sol) == sol[:, x_idx]
@test get_mix_arr(sol) == vcat.(sol[:, x_idx], sol[:, y_idx])
@test get_mix_tuple(sol) == tuple.(sol[:, x_idx], sol[:, y_idx])

# accessing parameters
@variables t x(t)
@parameters tau
D = Differential(t)

@named fol = ODESystem([D(x) ~ (1 - x) / tau])
prob = ODEProblem(fol, [x => 0.0], (0.0, 10.0), [tau => 3.0])
sol = solve(prob, Tsit5())
@test getp(fol, tau)(sol) == 3

@testset "OptimizationSolution" begin
    @variables begin
        x, [bounds = (-2.0, 2.0)]
        y, [bounds = (-1.0, 3.0)]
    end
    @parameters a=1 b=1
    loss = (a - x)^2 + b * (y - x^2)^2
    @named sys = OptimizationSystem(loss, [x, y], [a, b])
    u0 = [x => 1.0
        y => 2.0]
    p = [a => 1.0
        b => 100.0]
    prob = OptimizationProblem(sys, u0, p, grad = true, hess = true)
    sol = solve(prob, GradientDescent())
    @test sol[x]≈1 atol=1e-3
    @test sol[y]≈1 atol=1e-3
    @test getp(sys, a)(sol) ≈ 1
    @test getp(sys, b)(sol) ≈ 100
    @test sol.ps[a] ≈ 1
    @test sol.ps[b] ≈ 100
end
