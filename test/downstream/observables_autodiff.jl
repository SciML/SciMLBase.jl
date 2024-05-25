using ModelingToolkit, OrdinaryDiffEq
using Zygote
using ModelingToolkit: t_nounits as t, D_nounits as D
import SymbolicIndexingInterface as SII
import SciMLStructures as SS
using ModelingToolkitStandardLibrary
import ModelingToolkitStandardLibrary as MSL

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
    du_ = [0.0, 1.0, 1.0, 1.0]
    du = [du_ for _ in sol.u]
    @test du == gs

    # Observable in a vector
    gs, = gradient(sol) do sol
        sum(sum.(sol[[sys.w, sys.x]]))
    end
    du_ = [0.0, 1.0, 1.0, 2.0]
    du = [du_ for _ in sol.u]
    @test du == gs
end

# DAE

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
            source, input_signal, ground, ampermeter
        ])
end

@testset "DAE Observable function AD" begin
    model = create_model()
    sys = structural_simplify(model)

    prob = ODEProblem(sys, [], (0.0, 1.0))
    sol = solve(prob, Rodas4())

    gs, = gradient(sol) do sol
        sum(sol[sys.ampermeter.i])
    end
    du_ = [0.2, 1.0]
    du = [du_ for _ in sol.u]
    @test gs == du
end

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
