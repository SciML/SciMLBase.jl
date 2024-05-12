using ModelingToolkit, OrdinaryDiffEq
using Zygote

@parameters σ ρ β
@variables x(t) y(t) z(t) w(t)

eqs = [D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z,
    w ~ x + y + z + 2 * β]

@mtkbuild sys = ODESystem(eqs, t)

u0 = [D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0]

p = [σ => 28.0,
    ρ => 10.0,
    β => 8 / 3]

tspan = (0.0, 100.0)
prob = ODEProblem(sys, u0, tspan, p, jac = true)
sol = solve(prob, Tsit5())

@testset "AutoDiff Observable Functions" begin
    gs, = gradient(sol) do sol
        sum(sol[sys.w])
    end
    du_ = [0., 1., 1., 1.]
    du = [du_ for _ = sol.u]
    @test du == gs.u
end
