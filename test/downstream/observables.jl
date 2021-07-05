using ModelingToolkit, OrdinaryDiffEq, Test

@parameters t σ ρ β
@variables x(t) y(t) z(t)
D = Differential(t)

eqs = [D(x) ~ σ*(y-x),
       D(y) ~ x*(ρ-z)-y,
       D(z) ~ x*y - β*z]

lorenz1 = ODESystem(eqs,name=:lorenz1)
lorenz2 = ODESystem(eqs,name=:lorenz2)

@variables a, α
@parameters γ
connections = [0 ~ lorenz1.x + lorenz2.y + a*γ,
               α ~ 2lorenz1.x + a*γ]
connected = structural_simplify(ODESystem(connections,t,[a,α],[γ],systems=[lorenz1,lorenz2]))

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
prob = ODEProblem(connected,u0,tspan,p)
sol = solve(prob,Rodas4())

@test sol[lorenz1.x] isa Vector
@test sol[lorenz1.x,2] isa Float64
@test sol[lorenz1.x,:] isa Vector
@test length(sol[lorenz1.x,1:5]) == 5
@test sol[α] isa Vector
@test sol[α,3] isa Float64
@test length(sol[α,5:10]) == 6

#=
using Plots
plot(sol,vars=(lorenz2.x,lorenz2.z))
plot(sol,vars=(α,lorenz2.z))
plot(sol,vars=(lorenz2.x,α))
plot(sol,vars=α)
plot(sol,vars=(t,α))
=#
