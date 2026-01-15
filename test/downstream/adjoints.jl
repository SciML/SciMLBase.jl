using ModelingToolkit, OrdinaryDiffEq, SymbolicIndexingInterface, Test
using ModelingToolkit: t_nounits as t, D_nounits as D

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
@variables x(t) y(t) z(t)

eqs = [
    D(x) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z,
]

@named lorenz1 = System(eqs, t)
@named lorenz2 = System(eqs, t)

@parameters γ
@variables a(t) α(t)
connections = [
    0 ~ lorenz1.x + lorenz2.y + a * γ,
    α ~ 2lorenz1.x + a * γ,
]
@mtkcompile sys = System(connections, t, [a, α], [γ], systems = [lorenz1, lorenz2])

u0 = [
    lorenz1.x => 1.0,
    lorenz1.y => 0.0,
    lorenz1.z => 0.0,
    lorenz2.x => 0.0,
    lorenz2.y => 1.0,
    lorenz2.z => 0.0,
]

p = [
    lorenz1.σ => 10.0,
    lorenz1.ρ => 28.0,
    lorenz1.β => 8 / 3,
    lorenz2.σ => 10.0,
    lorenz2.ρ => 28.0,
    lorenz2.β => 8 / 3,
    γ => 2.0,
]

tspan = (0.0, 100.0)
prob = ODEProblem(sys, [u0; p], tspan)
sol = solve(prob, Rodas4())

@testset "Symbolic indexing gradients" begin
    for backend in REVERSE_BACKENDS
        @testset "$(backend_name(backend))" begin
            gs_sym = DifferentiationInterface.gradient(sol -> sum(sol[lorenz1.x]), backend, sol)
            idx_sym = SymbolicIndexingInterface.variable_index(sys, lorenz1.x)
            true_grad_sym = zeros(length(ModelingToolkit.unknowns(sys)))
            true_grad_sym[idx_sym] = 1.0

            @test all(map(x -> x == true_grad_sym, gs_sym))
        end
    end
end

@testset "Vector symbolic indexing gradients" begin
    for backend in REVERSE_BACKENDS
        @testset "$(backend_name(backend))" begin
            gs_vec = DifferentiationInterface.gradient(sol -> sum(sum.(sol[[lorenz1.x, lorenz2.x]])), backend, sol)
            idx_vecsym = SymbolicIndexingInterface.variable_index.(Ref(sys), [lorenz1.x, lorenz2.x])
            true_grad_vecsym = zeros(length(ModelingToolkit.unknowns(sys)))
            true_grad_vecsym[idx_vecsym] .= 1.0

            @test all(map(x -> x == true_grad_vecsym, gs_vec.u))
        end
    end
end

@testset "Tuple symbolic indexing gradients" begin
    for backend in REVERSE_BACKENDS
        @testset "$(backend_name(backend))" begin
            gs_tup = DifferentiationInterface.gradient(sol -> sum(sum.(collect.(sol[(lorenz1.x, lorenz2.x)]))), backend, sol)
            idx_tupsym = SymbolicIndexingInterface.variable_index.(Ref(sys), [lorenz1.x, lorenz2.x])
            true_grad_tupsym = zeros(length(ModelingToolkit.unknowns(sys)))
            true_grad_tupsym[idx_tupsym] .= 1.0

            @test all(map(x -> x == true_grad_tupsym, gs_tup.u))
        end
    end
end

@testset "Time series symbolic indexing gradients" begin
    for backend in REVERSE_BACKENDS
        @testset "$(backend_name(backend))" begin
            gs_ts = DifferentiationInterface.gradient(sol -> sum(sum.(sol[[lorenz1.x, lorenz2.x], :])), backend, sol)
            idx_vecsym = SymbolicIndexingInterface.variable_index.(Ref(sys), [lorenz1.x, lorenz2.x])
            true_grad_vecsym = zeros(length(ModelingToolkit.unknowns(sys)))
            true_grad_vecsym[idx_vecsym] .= 1.0

            @test all(map(x -> x == true_grad_vecsym, gs_ts.u))
        end
    end
end

# BatchedInterface AD
@variables x(t) = 1.0 y(t) = 1.0 z(t) = 1.0 w(t) = 1.0
@named sys1 = System([D(x) ~ x + y, D(y) ~ y * z, D(z) ~ z * t * x], t)
sys1 = complete(sys1)
prob1 = ODEProblem(sys1, [], (0.0, 10.0))
@named sys2 = System([D(x) ~ x + w, D(y) ~ w * t, D(w) ~ x + y + w], t)
sys2 = complete(sys2)
prob2 = ODEProblem(sys2, [], (0.0, 10.0))

bi = BatchedInterface((sys1, [x, y, z]), (sys2, [x, y, w]))
getter = getsym(bi)

@testset "BatchedInterface AD" begin
    for backend in REVERSE_BACKENDS
        @testset "$(backend_name(backend))" begin
            # Compute gradient with respect to prob1
            p1grad = DifferentiationInterface.gradient(prob1 -> sum(getter(prob1, prob2)), backend, prob1)
            # Compute gradient with respect to prob2
            p2grad = DifferentiationInterface.gradient(prob2 -> sum(getter(prob1, prob2)), backend, prob2)

            @test p1grad.u0 ≈ ones(3)
            testp2grad = zeros(3)
            testp2grad[variable_index(prob2, w)] = 1.0
            @test p2grad.u0 ≈ testp2grad
        end
    end
end
