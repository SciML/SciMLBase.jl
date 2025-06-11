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

@mtkcompile sys = System(eqs, t)

u0 = [D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0]

p = [σ => 28.0,
    ρ => 10.0,
    β => 8 / 3]

tspan = (0.0, 100.0)
prob = ODEProblem(sys, [u0; p], tspan)
sol = solve(prob, Tsit5())

@testset "AutoDiff Observable Functions" begin
    gs, = gradient(sol) do sol
        sum(sol[sys.w])
    end
    du_ = [1.0, 1.0, 1.0, 0.0]
    du = [du_ for _ in sol.u]
    @test du == gs.u

    # Observable in a vector
    gs, = gradient(sol) do sol
        sum(sum.(sol[[sys.w, sys.x]]))
    end
    du_ = [1.0, 1.0, 2.0, 0.0]
    du = [du_ for _ in sol.u]
    @test du == gs.u
end

@testset "AD Observable Functions for Initialization" begin
    iprob = prob.f.initialization_data.initializeprob
    isol = solve(iprob)
    gs, = Zygote.gradient(isol) do isol
        isol[w]
    end

    @test gs isa NamedTuple
    @test isempty(setdiff(fieldnames(typeof(gs)), fieldnames(typeof(isol))))

    # Compare gradient for parameters match from observed function
    # to ensure parameter gradients are passed through the observed function
    f = SII.observed(iprob.f.sys, w)
    gu0, gp = gradient(SII.state_values(iprob), SII.parameter_values(iprob)) do u0, p
        f(u0, p)
    end

    @test gs.prob.p == gp
end

# DAE

function create_model(; C₁ = 3e-5, C₂ = 1e-6)
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

    @named circuit_model = System(eqs, t,
        systems = [
            resistor1, resistor2, capacitor1, capacitor2,
            source, input_signal, ground, ampermeter
        ], defaults = [resistor1.n.v => 0.0])
end

@testset "DAE Observable function AD" begin
    model = create_model()
    sys = mtkcompile(model)

    prob = ODEProblem(sys, [], (0.0, 1.0))
    sol = solve(prob, Rodas4())

    gs, = gradient(sol) do sol
        sum(sol[sys.ampermeter.i])
    end
    du_ = [0.2, 1.0]
    du = [du_ for _ in sol.u]
    @test gs.u == du

    @testset "DAE Initialization Observable function AD" begin
        iprob = prob.f.initialization_data.initializeprob
        isol = solve(iprob)
        tunables, repack, _ = SS.canonicalize(SS.Tunable(), SII.parameter_values(iprob))
        gs, = gradient(isol) do isol
            isol[sys.ampermeter.i]
        end
        gt = gs.prob.p.tunable
        @test length(findall(!iszero, gt)) == 1
    end
end

@testset "Adjoints with DAE" begin
    model = create_model()
    sys = mtkcompile(model)
    prob = ODEProblem(sys, [], (0.0, 1.0))
    tunables, _, _ = SS.canonicalize(SS.Tunable(), prob.p)

    gs_mtkp, gs_p_new = gradient(prob.p, tunables) do p, new_tunables
        new_p = SS.replace(SS.Tunable(), p, new_tunables)
        new_prob = remake(prob, p = new_p)
        sol = solve(new_prob, Rodas4())
        sum(sol[sys.ampermeter.i])
    end

    @test isnothing(gs_mtkp)
    @test !isnothing(gs_p_new)
    @test length(gs_p_new) == length(tunables)
end
