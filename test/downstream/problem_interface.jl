using ModelingToolkit, OrdinaryDiffEq, Test

@parameters σ ρ β
@variables t x(t) y(t) z(t)
D = Differential(t)

eqs = [D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z]

@named sys = ODESystem(eqs)

sys = structural_simplify(sys)

u0 = [D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0]

p = [σ => 28.0,
    ρ => 10.0,
    β => 8 / 3]

tspan = (0.0, 100.0)
prob = ODEProblem(sys, u0, tspan, p, jac = true)

@test prob[σ] == 28.0
@test prob[ρ] == 10.0
@test prob[β] == 8 / 3

@test prob[x] == 1.0
@test prob[y] == 0.0
@test prob[z] == 0.0

prob[y] = 1.0
@test prob[y] == 1.0

prob[σ] = 1.0
@test prob[σ] == 1.0
