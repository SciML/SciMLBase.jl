using ModelingToolkit, OrdinaryDiffEq
using LinearAlgebra: dot
using OrdinaryDiffEqRosenbrock: Rodas5
using ModelingToolkit: t_nounits as t, D_nounits as D
import SymbolicIndexingInterface as SII
import SciMLStructures as SS
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
const ZYGOTE_BACKENDS = VERSION < v"1.12" ? [AutoZygote()] : []
const MOONCAKE_BACKENDS = [AutoMooncake()]

function backend_name(backend::ADTypes.AbstractADType)
    return string(typeof(backend).name.name)
end

# Mooncake's gradient for a struct returns a `Mooncake.Tangent` whose fields
# are stored under `.fields`, while Zygote returns a NamedTuple with direct
# property access. Normalize so both backends can be tested with the same
# accessor.
_unwrap_grad(gs) = hasproperty(gs, :fields) ? getfield(gs, :fields) : gs

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
    # Keep symbolic metadata outside AD while differentiating the numeric indexing result.
    observable_symbols = [sys.w, sys.x]
    for backend in ZYGOTE_BACKENDS
        @testset "$(backend_name(backend))" begin
            gs = DifferentiationInterface.gradient(sol -> sum(sol[sys.w]), backend, sol)
            du_ = [1.0, 1.0, 1.0, 0.0]
            du = [du_ for _ in sol[[D(x), x, y, z]]]
            @test du == gs.u

            gs2 = DifferentiationInterface.gradient(
                sol -> sum(sum.(sol[observable_symbols])), backend, sol
            )
            du_ = [1.0, 1.0, 2.0, 0.0]
            du = [du_ for _ in sol[[D(x), x, y, z]]]
            @test du == gs2.u
        end
    end
    for backend in MOONCAKE_BACKENDS
        @testset "$(backend_name(backend))" begin
            gs = DifferentiationInterface.gradient(
                sol -> sum(sol[sys.w]), backend, sol
            )
            du_ = [1.0, 1.0, 1.0, 0.0]
            du = [du_ for _ in sol[[D(x), x, y, z]]]
            @test du == _unwrap_grad(gs).u

            gs2 = DifferentiationInterface.gradient(
                sol -> sum(sum.(sol[observable_symbols])), backend, sol
            )
            du_v = [1.0, 1.0, 2.0, 0.0]
            duv = [du_v for _ in sol[[D(x), x, y, z]]]
            @test duv == _unwrap_grad(gs2).u
        end
    end
end

# `w` is not required by the generated initialization subsystem. `x` is retained
# there as the explicit observable `x ~ Initial(x)` and exercises the same rule.
@testset "AD Observable Functions for Initialization" begin
    for backend in ZYGOTE_BACKENDS
        @testset "$(backend_name(backend))" begin
            iprob = prob.f.initialization_data.initializeprob
            isol = solve(iprob)
            gs = DifferentiationInterface.gradient(isol -> isol[sys.x], backend, isol)

            @test gs isa NamedTuple
            @test isempty(setdiff(fieldnames(typeof(gs)), fieldnames(typeof(isol))))

            # Compare gradient for parameters match from observed function
            # to ensure parameter gradients are passed through the observed function
            f = SII.observed(iprob, sys.x)
            gp = DifferentiationInterface.gradient(
                p -> f(SII.state_values(iprob), p), backend, SII.parameter_values(iprob)
            )

            @test gs.prob.p == gp
        end
    end
    for backend in MOONCAKE_BACKENDS
        @testset "$(backend_name(backend))" begin
            iprob = prob.f.initialization_data.initializeprob
            isol = solve(iprob)
            gs = DifferentiationInterface.gradient(
                isol -> isol[sys.x], backend, isol
            )
            @test gs isa Mooncake.Tangent

            f = SII.observed(iprob, sys.x)
            p = SII.parameter_values(iprob)
            tun, repack, _ = SS.canonicalize(SS.Tunable(), p)
            gp_ref = ForwardDiff.gradient(tun) do t
                f(SII.state_values(iprob), repack(t))
            end
            @test count(!iszero, gp_ref) == 1
            @test only(filter(!iszero, gp_ref)) == 1.0
            @test _unwrap_grad(gs).prob.fields.p.fields.tunable ≈ gp_ref
        end
    end
end

# ============================================================
# Minimal DAE (replaces the MTSL electrical-circuit tests)
# ============================================================
# One differential equation, one algebraic equation, and one observed
# equation, where the observed variable s_dae appears directly in the
# algebraic equation.
#
#   D(u_dae) = -a_dae * u_dae + v_dae   (differential)
#   0 = v_dae - b_dae * s_dae            (algebraic — references observed s_dae)
#   s_dae = u_dae^2                      (observed)
#
# After @mtkcompile, both v_dae and s_dae become observed functions of the
# single differential state u_dae — exercising the same AD-through-observed
# codepath that the original MTSL tests covered.

@parameters a_dae b_dae
@variables u_dae(t) v_dae(t) s_dae(t)

dae_eqs = [
    D(u_dae) ~ -a_dae * u_dae + v_dae,
    0 ~ v_dae^2 - b_dae * s_dae,
    s_dae ~ 2u_dae + v_dae,
]

@mtkcompile simple_dae = System(dae_eqs, t)

prob_dae = ODEProblem(
    simple_dae,
    [u_dae => 1.0, a_dae => 2.0, b_dae => 0.5],
    (0.0, 1.0); guesses = [v_dae => 1.0]
)
sol_dae = solve(prob_dae, Rodas5())

@testset "DAE Observable function AD" begin
    # MTK may emit either `s = 2u + v` or the constraint-equivalent `s = v²/b`.
    # Their off-manifold gradients differ, but both have derivative `4v` along
    # the constraint tangent `(dv, du) = (2b, 2v - b)`.
    b_value = 0.5
    v_values = sol_dae[simple_dae.v_dae]
    v_index = SII.variable_index(sol_dae, simple_dae.v_dae)
    u_index = SII.variable_index(sol_dae, simple_dae.u_dae)
    expected_tangent_derivatives = 4 .* v_values

    function tangent_derivatives(gradients)
        return map(eachindex(gradients)) do k
            tangent = zeros(length(gradients[k]))
            tangent[v_index] = 2 * b_value
            tangent[u_index] = 2 * v_values[k] - b_value
            dot(gradients[k], tangent)
        end
    end

    @test sol_dae[simple_dae.s_dae] ≈
        2 .* sol_dae[simple_dae.u_dae] .+ v_values
    for backend in ZYGOTE_BACKENDS
        @testset "$(backend_name(backend))" begin
            gs = DifferentiationInterface.gradient(
                sol -> sum(sol[simple_dae.s_dae]), backend, sol_dae
            )
            @test tangent_derivatives(gs.u) ≈ expected_tangent_derivatives
        end
    end
    for backend in MOONCAKE_BACKENDS
        @testset "$(backend_name(backend))" begin
            gs = DifferentiationInterface.gradient(
                sol -> sum(sol[simple_dae.s_dae]), backend, sol_dae
            )
            @test tangent_derivatives(_unwrap_grad(gs).u) ≈
                expected_tangent_derivatives
        end
    end

    @testset "DAE Initialization Observable function AD" begin
        # `s_dae` can be eliminated from the generated initialization subsystem;
        # `u_dae ~ Initial(u_dae)` is an observable that the subsystem exposes.
        for backend in ZYGOTE_BACKENDS
            @testset "$(backend_name(backend))" begin
                iprob = prob_dae.f.initialization_data.initializeprob
                isol = solve(iprob)
                gs = DifferentiationInterface.gradient(
                    isol -> isol[simple_dae.u_dae], backend, isol
                )
                gt = gs.prob.p.tunable
                @test length(findall(!iszero, gt)) == 1
            end
        end
        # Mooncake fails while copying the nonempty initialization-solution tangent.
        for backend in MOONCAKE_BACKENDS
            @testset "$(backend_name(backend)) (broken)" begin
                @test_broken begin
                    iprob = prob_dae.f.initialization_data.initializeprob
                    isol = solve(iprob)
                    gs = DifferentiationInterface.gradient(
                        isol -> isol[simple_dae.u_dae], backend, isol
                    )
                    gt = _unwrap_grad(gs).prob.fields.p.fields.tunable
                    length(findall(!iszero, gt)) == 1 &&
                        only(filter(!iszero, gt)) == 1.0
                end
            end
        end
    end
end

@testset "Adjoints with DAE" begin
    tunables_dae, _, _ = SS.canonicalize(SS.Tunable(), prob_dae.p)

    function loss_dae(new_tunables)
        new_p = SS.replace(SS.Tunable(), prob_dae.p, new_tunables)
        new_prob = remake(prob_dae; p = new_p)
        # `prob_dae` carries a non-trivial mass matrix (DAE), so the solver
        # must be DAE-capable. The original test used `Rodas4()`; the recent
        # AI rewrite (d3fa810) inadvertently swapped in `Tsit5()`, which
        # raises `"This solver is not able to use mass matrices"` immediately
        # in `_ode_init`. Match `prob_dae`'s top-level `solve(..., Rodas5())`.
        sol = solve(new_prob, Rodas5())
        return sum(sol[simple_dae.s_dae])
    end

    # Zygote DAE adjoints (previously broken via the ChainRules `_solve_adjoint`
    # path evaluating the initialization SCCNonlinearProblem's observable
    # `TimeIndependentObservedFunction` with `u = nothing`; see
    # https://github.com/SciML/SciMLBase.jl/issues/1233) now return a gradient
    # of the expected length.
    for backend in ZYGOTE_BACKENDS
        @testset "$(backend_name(backend))" begin
            gs = DifferentiationInterface.gradient(loss_dae, backend, tunables_dae)
            @test !isnothing(gs)
            @test length(gs) == length(tunables_dae)
        end
    end
    # Mooncake handles this adjoint through its own `build_rrule` /
    # `getindex` primitive and returns a gradient of the expected length.
    for backend in MOONCAKE_BACKENDS
        @testset "$(backend_name(backend))" begin
            gs = DifferentiationInterface.gradient(loss_dae, backend, tunables_dae)
            @test !isnothing(gs)
            @test length(gs) == length(tunables_dae)
        end
    end
end

@testset "NonlinearSolution scalar getindex under AD (SciMLSensitivity#1446)" begin
    using SteadyStateDiffEq
    using OrdinaryDiffEqRosenbrock: Rodas5

    # `save_idxs = i::Integer` makes `solve` return a NonlinearSolution{T,0,T,...}
    # whose `u::Float64` is a scalar (vs `save_idxs = i:i` which keeps a 1-d
    # Vector{T}). Indexing the scalar solution as `sol[1]` returns that scalar.
    # Before the dedicated `Integer` rrule on `AbstractNonlinearSolution`, the
    # generic symbolic-indexing rrule silently dropped the cotangent because
    # `sol.u`'s fdata for a scalar field is `NoFData`, not a `Vector{<:Real}`.
    function f_iip!(du, u, p, t)
        du[1] = p[1] + p[2] * u[1]
        du[2] = p[3] * u[1] + p[4] * u[2]
        return nothing
    end
    u0_ss = zeros(2)
    p_ss = [2.0, -2.0, 1.0, -4.0]
    prob_ss = SteadyStateProblem(f_iip!, u0_ss, p_ss)

    # Tight solve tolerances so every gradient below is resolved to the true
    # analytic value rather than to the steady-state solver's default error.
    # The loss is `u₁` at steady state, where `p₁ + p₂ u₁ = 0`, so `u₁ = -p₁/p₂`
    # and the exact gradient is `[-1/p₂, p₁/p₂², 0, 0] = [0.5, 0.5, 0, 0]`. At the
    # default `DynamicSS` tolerance the ForwardDiff reference is only accurate to
    # ~1.2e-6 — larger than the `rtol = 1e-6` compared against below — so the test
    # was brittle to AD-backend accuracy drift. `abstol = reltol = 1e-10` brings
    # every solve to ~1e-9 of the true gradient, well inside the tolerance.
    ss_tol = (; abstol = 1.0e-10, reltol = 1.0e-10)

    # Reference gradient: the same loss computed via a path that does not
    # touch the buggy dispatch (full solve + index into the resulting array).
    ref_loss = θ -> sum(
        Array(
            solve(
                prob_ss, DynamicSS(Rodas5()); u0 = u0_ss, p = θ,
                sensealg = SteadyStateAdjoint(), ss_tol...,
            )
        )[1]
    )
    grad_ref = DifferentiationInterface.gradient(ref_loss, AutoForwardDiff(), p_ss)

    # Scalar `save_idxs` + `[1]` — this is the dispatch that used to return zero.
    scalar_loss_ssa = θ -> solve(
        prob_ss, DynamicSS(Rodas5()); u0 = u0_ss, p = θ,
        save_idxs = 1, sensealg = SteadyStateAdjoint(), ss_tol...,
    )[1]
    scalar_loss_default = θ -> solve(
        prob_ss, DynamicSS(Rodas5()); u0 = u0_ss, p = θ, save_idxs = 1, ss_tol...,
    )[1]

    for backend in MOONCAKE_BACKENDS
        @testset "$(backend_name(backend))" begin
            grad_ssa = DifferentiationInterface.gradient(scalar_loss_ssa, backend, p_ss)
            @test grad_ssa ≈ grad_ref rtol = 1.0e-6
            grad_def = DifferentiationInterface.gradient(scalar_loss_default, backend, p_ss)
            @test grad_def ≈ grad_ref rtol = 1.0e-6
        end
    end
end

@testset "Integer time-step indexing under AD (issue #1325)" begin
    function lotka_volterra!(du, u, p, t)
        x, y = u
        du[1] = (p[1] - p[2] * y) * x
        du[2] = (p[4] * x - p[3]) * y
    end

    lv_prob = ODEProblem(
        lotka_volterra!, [1.0, 1.0], (0.0, 10.0), [1.5, 1.0, 3.0, 1.0]
    )

    function lv_logp(θ)
        predicted = solve(
            lv_prob, Tsit5(); p = θ, saveat = 0.1, abstol = 1.0e-6, reltol = 1.0e-6
        )
        lp = 0.0
        for i in eachindex(predicted)
            lp += sum(predicted[i])
        end
        return lp
    end

    θ0 = [1.5, 1.0, 3.0, 1.0]
    grad_fd = DifferentiationInterface.gradient(lv_logp, AutoForwardDiff(), θ0)

    for backend in MOONCAKE_BACKENDS
        @testset "$(backend_name(backend))" begin
            grad = DifferentiationInterface.gradient(lv_logp, backend, θ0)
            @test grad ≈ grad_fd rtol = 1.0e-4
        end
    end
    for backend in ZYGOTE_BACKENDS
        @testset "$(backend_name(backend))" begin
            grad = DifferentiationInterface.gradient(lv_logp, backend, θ0)
            @test grad ≈ grad_fd rtol = 1.0e-4
        end
    end
end
