using ModelingToolkit, OrdinaryDiffEq, Test

@parameters t σ ρ β
@variables x(t) y(t) z(t)
D = Differential(t)

eqs = [D(x) ~ σ*(y-x),
       D(y) ~ x*(ρ-z)-y,
       D(z) ~ x*y - β*z]

lorenz1 = ODESystem(eqs,name=:lorenz1)
lorenz2 = ODESystem(eqs,name=:lorenz2)

@parameters γ
@variables a(t) α(t)
connections = [0 ~ lorenz1.x + lorenz2.y + a*γ,
               α ~ 2lorenz1.x + a*γ]
sys = ODESystem(connections,t,[a,α],[γ],systems=[lorenz1,lorenz2])
sys_simplified = structural_simplify(sys)

u0 = [lorenz1.x => 1.0,
      lorenz1.y => 0.0,
      lorenz1.z => 0.0,
      lorenz2.x => 0.0,
      lorenz2.y => 1.0,
      lorenz2.z => 0.0,
      a => 2.0]

p  = [lorenz1.σ => 10.0,
      lorenz1.ρ => 28.0,
      lorenz1.β => 8/3,
      lorenz2.σ => 10.0,
      lorenz2.ρ => 28.0,
      lorenz2.β => 8/3,
      γ => 2.0]

tspan = (0.0,100.0)
prob = ODEProblem(sys_simplified,u0,tspan,p)
sol = solve(prob,Rodas4())

@test sol[lorenz1.x] isa Vector
@test sol[lorenz1.x,2] isa Float64
@test sol[lorenz1.x,:] isa Vector
@test sol[t] isa Vector
@test sol[t,2] isa Float64
@test sol[t,:] isa Vector
@test length(sol[lorenz1.x,1:5]) == 5
@test sol[α] isa Vector
@test sol[α,3] isa Float64
@test length(sol[α,5:10]) == 6

# Check if indexing using variable names from interpolated solution works
interpolated_sol = sol(0.0:1.0:10.0)
@test interpolated_sol[α] isa Vector
@test interpolated_sol[α,:] isa Vector
@test interpolated_sol[α,2] isa Float64
@test length(interpolated_sol[α,1:5]) == 5
@test interpolated_sol[α] ≈ 2interpolated_sol[lorenz1.x] .+ interpolated_sol[a].*2.0
@test collect(interpolated_sol[t]) isa Vector
@test collect(interpolated_sol[t,:]) isa Vector
@test interpolated_sol[t,2] isa Float64
@test length(interpolated_sol[t,1:5]) == 5


sol1 = sol(0.0:1.0:10.0)
@test sol1.u isa Vector
@test first(sol1.u) isa Vector
@test length(sol1.u) == 11
@test length(sol1.t) == 11

sol2 = sol(0.1)
@test sol2 isa Vector
@test length(sol2) == length(states(sys_simplified))
@test first(sol2) isa Real

sol3 = sol(0.0:1.0:10.0, idxs=[lorenz1.x, lorenz2.x])
@test sol3.u isa Vector
@test first(sol3.u) isa Vector
@test length(sol3.u) == 11
@test length(sol3.t) == 11
@test collect(sol3[t]) ≈ sol3.t
@test collect(sol3[t, 1:5]) ≈ sol3.t[1:5]
@test_throws ErrorException sol(0.0:1.0:10.0, idxs=[lorenz1.x, 1])

sol4 = sol(0.1, idxs=[lorenz1.x, lorenz2.x])
@test sol4 isa Vector
@test length(sol4) == 2
@test first(sol4) isa Real
@test_throws ErrorException sol(0.1, idxs=[lorenz1.x, 1])

sol5 = sol(0.0:1.0:10.0, idxs=lorenz1.x)
@test sol5.u isa Vector
@test first(sol5.u) isa Real
@test length(sol5.u) == 11
@test length(sol5.t) == 11
@test collect(sol5[t]) ≈ sol3.t
@test collect(sol5[t, 1:5]) ≈ sol3.t[1:5]
@test_throws ErrorException sol(0.0:1.0:10.0, idxs=1.2)

sol6 = sol(0.1, idxs=lorenz1.x)
@test sol6 isa Real
@test_throws ErrorException sol(0.1, idxs=1.2)

sol7 = sol(0.0:1.0:10.0, idxs=[2,1])
@test sol7.u isa Vector
@test first(sol7.u) isa Vector
@test length(sol7.u) == 11
@test length(sol7.t) == 11
@test collect(sol7[t]) ≈ sol3.t
@test collect(sol7[t, 1:5]) ≈ sol3.t[1:5]

sol8 = sol(0.1, idxs=[2,1])
@test sol8 isa Vector
@test length(sol8) == 2
@test first(sol8) isa Real

sol9 = sol(0.0:1.0:10.0, idxs=2)
@test sol9.u isa Vector
@test first(sol9.u) isa Real
@test length(sol9.u) == 11
@test length(sol9.t) == 11
@test collect(sol9[t]) ≈ sol3.t
@test collect(sol9[t, 1:5]) ≈ sol3.t[1:5]

sol10 = sol(0.1, idxs=2)
@test sol10 isa Real


#=
using Plots
plot(sol,vars=(lorenz2.x,lorenz2.z))
plot(sol,vars=(α,lorenz2.z))
plot(sol,vars=(lorenz2.x,α))
plot(sol,vars=α)
plot(sol,vars=(t,α))
=#
