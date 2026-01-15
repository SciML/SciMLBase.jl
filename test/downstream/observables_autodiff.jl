using ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D
import SymbolicIndexingInterface as SII
import SciMLStructures as SS
using ModelingToolkitStandardLibrary
import ModelingToolkitStandardLibrary as MSL
using SciMLSensitivity
using Test

# DifferentiationInterface with version-dependent backends
using DifferentiationInterface
using ADTypes
using ForwardDiff: ForwardDiff
using Mooncake: Mooncake
if VERSION < v"1.12"
    using Zygote: Zygote
    using Enzyme: Enzyme
end

# Define available reverse-mode backends based on Julia version
const REVERSE_BACKENDS = if VERSION < v"1.12"
    [AutoZygote(), AutoMooncake()]
else
    [AutoMooncake()]
end

function backend_name(backend::ADTypes.AbstractADType)
    return string(typeof(backend).name.name)
end

@parameters σ ρ β
@variables x(t) y(t) z(t) w(t)

eqs = [
    D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z,
    w ~ x + y + z + 2 * β,
]

@mtkcompile sys = System(eqs, t)

u0 = [
    D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0,
]

p = [
    σ => 28.0,
    ρ => 10.0,
    β => 8 / 3,
]

tspan = (0.0, 100.0)
prob = ODEProblem(sys, [u0; p], tspan)
sol = solve(prob, Tsit5())

@testset "AutoDiff Observable Functions" begin
    for backend in REVERSE_BACKENDS
        @testset "$(backend_name(backend))" begin
            gs = DifferentiationInterface.gradient(sol -> sum(sol[sys.w]), backend, sol)
            du_ = [1.0, 1.0, 1.0, 0.0]
            du = [du_ for _ in sol[[D(x), x, y, z]]]
            @test du == gs.u

            # Observable in a vector
            gs2 = DifferentiationInterface.gradient(sol -> sum(sum.(sol[[sys.w, sys.x]])), backend, sol)
            du_ = [1.0, 1.0, 2.0, 0.0]
            du = [du_ for _ in sol[[D(x), x, y, z]]]
            @test du == gs2.u
        end
    end
end

@testset "AD Observable Functions for Initialization" begin
    for backend in REVERSE_BACKENDS
        @testset "$(backend_name(backend))" begin
            iprob = prob.f.initialization_data.initializeprob
            isol = solve(iprob)
            gs = DifferentiationInterface.gradient(isol -> isol[w], backend, isol)

            @test gs isa NamedTuple
            @test isempty(setdiff(fieldnames(typeof(gs)), fieldnames(typeof(isol))))

            # Compare gradient for parameters match from observed function
            # to ensure parameter gradients are passed through the observed function
            f = SII.observed(iprob.f.sys, w)
            gu0 = DifferentiationInterface.gradient(u0 -> f(u0, SII.parameter_values(iprob)), backend, SII.state_values(iprob))
            gp = DifferentiationInterface.gradient(p -> f(SII.state_values(iprob), p), backend, SII.parameter_values(iprob))

            @test gs.prob.p == gp
        end
    end
end

# DAE

function create_model(; C₁ = 3.0e-5, C₂ = 1.0e-6)
    @named resistor1 = MSL.Electrical.Resistor(R = 5.0)
    @named resistor2 = MSL.Electrical.Resistor(R = 2.0)
    @named capacitor1 = MSL.Electrical.Capacitor(C = C₁)
    @named capacitor2 = MSL.Electrical.Capacitor(C = C₂)
    @named source = MSL.Electrical.Voltage()
    @named input_signal = MSL.Blocks.Sine(frequency = 100.0)
    @named ground = MSL.Electrical.Ground()
    @named ampermeter = MSL.Electrical.CurrentSensor()

    eqs = [
        connect(input_signal.output, source.V)
        connect(source.p, capacitor1.n, capacitor2.n)
        connect(source.n, resistor1.p, resistor2.p, ground.g)
        connect(resistor1.n, capacitor1.p, ampermeter.n)
        connect(resistor2.n, capacitor2.p, ampermeter.p)
    ]

    return @named circuit_model = System(
        eqs, t,
        systems = [
            resistor1, resistor2, capacitor1, capacitor2,
            source, input_signal, ground, ampermeter,
        ], defaults = [resistor1.n.v => 0.0]
    )
end

@testset "DAE Observable function AD" begin
    model = create_model()
    sys = mtkcompile(model)

    prob = ODEProblem(sys, [], (0.0, 1.0))
    sol = solve(prob, Rodas4())

    for backend in REVERSE_BACKENDS
        @testset "$(backend_name(backend))" begin
            gs = DifferentiationInterface.gradient(sol -> sum(sol[sys.ampermeter.i]), backend, sol)
            du_ = [0.2, 1.0]
            du = [du_ for _ in sol.u]
            @test gs.u == du
        end
    end

    @testset "DAE Initialization Observable function AD" begin
        for backend in REVERSE_BACKENDS
            @testset "$(backend_name(backend))" begin
                iprob = prob.f.initialization_data.initializeprob
                isol = solve(iprob)
                tunables, repack, _ = SS.canonicalize(SS.Tunable(), SII.parameter_values(iprob))
                gs = DifferentiationInterface.gradient(isol -> isol[sys.ampermeter.i], backend, isol)
                gt = gs.prob.p.tunable
                @test length(findall(!iszero, gt)) == 1
            end
        end
    end
end

@testset "Adjoints with DAE" begin
    model = create_model()
    sys = mtkcompile(model)
    prob = ODEProblem(sys, [], (0.0, 1.0))
    tunables, _, _ = SS.canonicalize(SS.Tunable(), prob.p)

    for backend in REVERSE_BACKENDS
        @testset "$(backend_name(backend))" begin
            # Need to compute gradients with respect to both p and new_tunables
            # For DifferentiationInterface, we compute each gradient separately
            function loss_wrt_tunables(new_tunables)
                new_p = SS.replace(SS.Tunable(), prob.p, new_tunables)
                new_prob = remake(prob, p = new_p)
                sol = solve(new_prob, Rodas4())
                sum(sol[sys.ampermeter.i])
            end

            gs_p_new = DifferentiationInterface.gradient(loss_wrt_tunables, backend, tunables)

            @test !isnothing(gs_p_new)
            @test length(gs_p_new) == length(tunables)
        end
    end
end
