using ModelingToolkit, OrdinaryDiffEq
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
    for backend in ZYGOTE_BACKENDS
        @testset "$(backend_name(backend))" begin
            gs = DifferentiationInterface.gradient(sol -> sum(sol[sys.w]), backend, sol)
            du_ = [1.0, 1.0, 1.0, 0.0]
            du = [du_ for _ in sol[[D(x), x, y, z]]]
            @test du == gs.u

            # Observable in a vector
            # Zygote returns incorrect gradient on Julia 1.10 (elements swapped)
            # See https://github.com/SciML/SciMLBase.jl/issues/1233
            gs2 = DifferentiationInterface.gradient(
                sol -> sum(sum.(sol[[sys.w, sys.x]])), backend, sol
            )
            du_ = [1.0, 1.0, 2.0, 0.0]
            du = [du_ for _ in sol[[D(x), x, y, z]]]
            @test_broken du == gs2.u
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

            # Vector observable not yet supported by the Mooncake getindex
            # primitive — falls back to differentiating the [...] AbstractArray
            # constructor and hits the dispatch chain.
            @test_broken begin
                gs2 = DifferentiationInterface.gradient(
                    sol -> sum(sum.(sol[[sys.w, sys.x]])), backend, sol
                )
                du_v = [1.0, 1.0, 2.0, 0.0]
                duv = [du_v for _ in sol[[D(x), x, y, z]]]
                duv == _unwrap_grad(gs2).u
            end
        end
    end
end

@testset "AD Observable Functions for Initialization" begin
    for backend in ZYGOTE_BACKENDS
        @testset "$(backend_name(backend))" begin
            iprob = prob.f.initialization_data.initializeprob
            isol = solve(iprob)
            gs = DifferentiationInterface.gradient(isol -> isol[w], backend, isol)

            @test gs isa NamedTuple
            @test isempty(setdiff(fieldnames(typeof(gs)), fieldnames(typeof(isol))))

            # Compare gradient for parameters match from observed function
            # to ensure parameter gradients are passed through the observed function
            f = SII.observed(iprob.f.sys, w)
            gu0 = DifferentiationInterface.gradient(
                u0 -> f(u0, SII.parameter_values(iprob)), backend, SII.state_values(iprob)
            )
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
            # Note: `isol[w]` with the bare global `w` trips Mooncake's
            # `__verify_const(::Num, ...)` check for non-const globals, which
            # uses `==` on a `Num` and hits the symbolic-in-bool error. That
            # is unrelated to the SciMLBase rrule — use `sys.w` (a fresh
            # getproperty) so the symbol isn't a GlobalRef, matching the
            # timeseries tests above.
            gs = DifferentiationInterface.gradient(
                isol -> isol[sys.w], backend, isol
            )
            @test gs isa Mooncake.Tangent

            # Compare the Mooncake parameter gradient against ForwardDiff
            # applied to the same inner observed function. Both should
            # agree exactly on a linear expression (`w ~ x + y + z + 2β`).
            f = SII.observed(iprob.f.sys, sys.w)
            p = SII.parameter_values(iprob)
            tun, repack, _ = SS.canonicalize(SS.Tunable(), p)
            gp_ref = ForwardDiff.gradient(tun) do t
                f(SII.state_values(iprob), repack(t))
            end
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
    # `@mtkcompile` substitutes the algebraic constraint
    # `0 ~ v_dae^2 - b_dae*s_dae` into the s_dae observable instead of using
    # the surface definition `s_dae ~ 2u_dae + v_dae`. With b_dae = 0.5 the
    # emitted observed function is `s_dae(u, p, t) = v^2 / b = 2v^2`, where
    # the state ordering is `[v_dae, u_dae]`. Both expressions agree on the
    # DAE manifold but their gradients in the embedding state space do not:
    # ∂(2v^2)/∂u_state = [4v, 0] (Mooncake and ForwardDiff both give this),
    # whereas the literal `2u + v` derivative would be [1, 2]. Assert
    # against the actually-emitted observable.
    du = [[4 * sol_dae.u[k][1], 0.0] for k in eachindex(sol_dae.u)]
    for backend in ZYGOTE_BACKENDS
        @testset "$(backend_name(backend))" begin
            gs = DifferentiationInterface.gradient(
                sol -> sum(sol[simple_dae.s_dae]), backend, sol_dae
            )
            @test gs.u ≈ du
        end
    end
    for backend in MOONCAKE_BACKENDS
        @testset "$(backend_name(backend))" begin
            gs = DifferentiationInterface.gradient(
                sol -> sum(sol[simple_dae.s_dae]), backend, sol_dae
            )
            @test _unwrap_grad(gs).u ≈ du
        end
    end

    @testset "DAE Initialization Observable function AD" begin
        for backend in ZYGOTE_BACKENDS
            @testset "$(backend_name(backend))" begin
                iprob = prob_dae.f.initialization_data.initializeprob
                isol = solve(iprob)
                tunables, repack, _ = SS.canonicalize(
                    SS.Tunable(), SII.parameter_values(iprob)
                )
                gs = DifferentiationInterface.gradient(
                    isol -> isol[simple_dae.s_dae], backend, isol
                )
                gt = gs.prob.p.tunable
                @test length(findall(!iszero, gt)) == 1
            end
        end
        # Mooncake does not support SymbolicIndexingInterface AD yet
        for backend in MOONCAKE_BACKENDS
            @testset "$(backend_name(backend)) (broken)" begin
                @test_broken begin
                    iprob = prob_dae.f.initialization_data.initializeprob
                    isol = solve(iprob)
                    gs = DifferentiationInterface.gradient(
                        isol -> isol[simple_dae.s_dae], backend, isol
                    )
                    gt = gs.prob.p.tunable
                    length(findall(!iszero, gt)) == 1
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

    # Zygote DAE adjoints currently broken. The ChainRules `_solve_adjoint`
    # path calls `get_concrete_problem` -> `get_updated_symbolic_problem`,
    # which evaluates the initialization SCCNonlinearProblem's observable
    # `TimeIndependentObservedFunction` with `u = nothing`, and the generated
    # `RuntimeGeneratedFunction` then hits
    # `MethodError: no method matching getindex(::Nothing, ::Int64)`.
    # See https://github.com/SciML/SciMLBase.jl/issues/1233
    for backend in ZYGOTE_BACKENDS
        @testset "$(backend_name(backend))" begin
            @test_broken begin
                gs = DifferentiationInterface.gradient(loss_dae, backend, tunables_dae)
                !isnothing(gs) && length(gs) == length(tunables_dae)
            end
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

    # Reference gradient: the same loss computed via a path that does not
    # touch the buggy dispatch (full solve + index into the resulting array).
    ref_loss = θ -> sum(
        Array(
            solve(
                prob_ss, DynamicSS(Rodas5()); u0 = u0_ss, p = θ,
                sensealg = SteadyStateAdjoint(),
            )
        )[1]
    )
    grad_ref = DifferentiationInterface.gradient(ref_loss, AutoForwardDiff(), p_ss)

    # Scalar `save_idxs` + `[1]` — this is the dispatch that used to return zero.
    scalar_loss_ssa = θ -> solve(
        prob_ss, DynamicSS(Rodas5()); u0 = u0_ss, p = θ,
        save_idxs = 1, sensealg = SteadyStateAdjoint(),
    )[1]
    scalar_loss_default = θ -> solve(
        prob_ss, DynamicSS(Rodas5()); u0 = u0_ss, p = θ, save_idxs = 1,
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
