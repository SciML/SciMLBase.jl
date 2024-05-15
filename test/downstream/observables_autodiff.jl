using ModelingToolkit, OrdinaryDiffEq
using Zygote
using ModelingToolkit: t_nounits as t, D_nounits as D
import SymbolicIndexingInterface as SII
import SciMLStructures as SS

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

# Lorenz

@parameters σ ρ β
@variables x(t) y(t) z(t)

eqs = [D(x) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z]

@named lorenz1 = ODESystem(eqs, t)
@named lorenz2 = ODESystem(eqs, t)

@parameters γ
@variables a(t) α(t)
connections = [0 ~ lorenz1.x + lorenz2.y + a * γ,
    α ~ 2lorenz1.x + a * γ]
@mtkbuild sys = ODESystem(connections, t, [a, α], [γ], systems = [lorenz1, lorenz2])

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
prob = ODEProblem(sys, u0, tspan, p)
integ = init(prob, Rodas4())
sol = solve(prob, Rodas4())

gt = reduce(hcat, sol[[sys.a, sys.α]]) .+ randn.()

gs, = Zygote.gradient(sol) do sol
    mean(abs.(sol[[sys.a, sys.α]] .- gt), dims = 2)
end

# DAE

using ModelingToolkit, OrdinaryDiffEq, Zygote
using ModelingToolkitStandardLibrary
import ModelingToolkitStandardLibrary as MSL
using SciMLStructures

function create_model(; C₁ = 3e-5, C₂ = 1e-6)
    @variables t
    @named resistor1 = MSL.Electrical.Resistor(R = 5.0)
    @named resistor2 = MSL.Electrical.Resistor(R = 2.0)
    @named capacitor1 = MSL.Electrical.Capacitor(C = C₁)
    @named capacitor2 = MSL.Electrical.Capacitor(C = C₂)
    @named source = MSL.Electrical.Voltage()
    @named input_signal = MSL.Blocks.Sine(frequency = 100.0)
    @named ground = MSL.Electrical.Ground()
    @named ampermeter = MSL.Electrical.CurrentSensor()

    eqs = [connect(input_signal.output, source.V)
        connect(source.p, capacitor1.n, capacitor2.n)
        connect(source.n, resistor1.p, resistor2.p, ground.g)
        connect(resistor1.n, capacitor1.p, ampermeter.n)
        connect(resistor2.n, capacitor2.p, ampermeter.p)]

    @named circuit_model = ODESystem(eqs, t,
        systems = [
            resistor1, resistor2, capacitor1, capacitor2,
            source, input_signal, ground, ampermeter,
        ])
end

model = create_model()
sys = structural_simplify(model)

prob = ODEProblem(sys, [], (0.0, 1.0))
sol = solve(prob, Rodas4())
pf = getp(sol, sys.resistor1.R)
mtkparams = SII.parameter_values(sol)
tunables, _, _ = SS.canonicalize(SS.Tunable(), mtkparams)
p_new = rand(length(tunables))

# @testset "Adjoints with DAE" begin
#     gs_mtkp, gs_p_new = gradient(mtkparams, p_new) do p, new_tunables
#         new_p = SciMLStructures.replace(SciMLStructures.Tunable(), p, new_tunables)
#         new_prob = remake(prob, p = new_p)
#         sol = solve(new_prob, Rodas4())
#         @show size(sol)
#         # mean(abs.(sol[sys.ampermeter.i] .- gt))
#         sum(sol[sys.ampermeter.i])
#     end
# 
#     @test isnothing(gs_mtkp)
#     @test length(gs_p_new) == length(p_new)
# end
