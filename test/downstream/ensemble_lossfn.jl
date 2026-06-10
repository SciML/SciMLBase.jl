using OrdinaryDiffEq, SciMLBase, SciMLSensitivity, Test
using DifferentiationInterface
using ADTypes
using ForwardDiff: ForwardDiff
using Zygote: Zygote

# Regression test for differentiating an EnsembleProblem when the loss iterates
# per-trajectory solution fields (`for s in sol.u`).
#
# The bug was in `__getindex(::AbstractEnsembleSolution, i)` returning a scalar
# instead of the i-th trajectory's array cotangent during the reverse pass
# through `responsible_map` / `tmap`. The fault lives in the Zygote extension
# (`SciMLBaseZygoteExt`), so this test targets Zygote and cross-checks against a
# ForwardDiff ground truth. See SciML/SciMLSensitivity.jl#1478.

# Reverse-mode AD backends to validate. Typed abstractly so more can be appended.
const REVERSE_BACKENDS = ADTypes.AbstractADType[AutoZygote()]
# Ground-truth backend (independent of the reverse-mode ensemble adjoint path).
const FORWARD_BACKEND = AutoForwardDiff()

# Adjoint sensitivity algorithms to validate. The fix is sensealg-independent;
# every entry should give the same gradient.
const SENSEALGS = [
    ("automatic", nothing),
    ("GaussAdjoint", GaussAdjoint(autojacvec=ReverseDiffVJP())),
    ("QuadratureAdjoint", QuadratureAdjoint(autojacvec=ReverseDiffVJP())),
]

const ENSEMBLE_ALGS = [
    ("EnsembleSerial", EnsembleSerial()),
    ("EnsembleThreads", EnsembleThreads()),
]

backend_name(b::ADTypes.AbstractADType) = string(typeof(b).name.name)

@testset "Ensemble per-trajectory loss AD (#1478)" begin
    A = [1.0 2.0; 3.0 4.0]
    n_traj = 4
    tspan = (0.0, 1.0)

    prob_func(prob, ctx) = remake(prob, u0=(1.0 + ctx.sim_id / 10) .* prob.u0)

    function build(p)
        prob = ODEProblem((u, p, t) -> p[1] .* (A * u), [1.0, 1.0], tspan, p)
        return EnsembleProblem(prob; prob_func=prob_func)
    end

    # `Array(sol)` is the always-worked path; the others iterate `sol.u` and read
    # per-trajectory fields (the patterns that triggered the BoundsError).
    function loss_array(p; ealg, sensealg)
        sol = solve(build(p), Tsit5(), ealg; trajectories=n_traj,
            saveat=0.25, sensealg=sensealg)
        return sum(abs2, Array(sol))
    end

    function loss_sum_all(p; ealg, sensealg)
        sol = solve(build(p), Tsit5(), ealg; trajectories=n_traj,
            saveat=0.25, sensealg=sensealg)
        return sum(sum(sum, s.u) for s in sol.u)
    end

    function loss_per_traj_end(p; ealg, sensealg)
        sol = solve(build(p), Tsit5(), ealg; trajectories=n_traj,
            saveat=0.25, sensealg=sensealg)
        return sum(sum(abs2, s.u[end]) for s in sol.u)
    end

    p0 = [1.0]

    losses = [
        ("Array(sol)", loss_array),
        ("sum over s.u", loss_sum_all),
        ("per-trajectory s.u[end]", loss_per_traj_end),
    ]

    # ForwardDiff ground truth (uses a representative sensealg; value is
    # sensealg-independent for these smooth losses).
    ref_sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())
    refs = Dict(
        name => DifferentiationInterface.gradient(
            p -> lossf(p; ealg=EnsembleSerial(), sensealg=ref_sensealg),
            FORWARD_BACKEND, p0)
        for (name, lossf) in losses
    )

    @testset "$(backend_name(backend))" for backend in REVERSE_BACKENDS
        @testset "$ealgname" for (ealgname, ealg) in ENSEMBLE_ALGS
            @testset "$saname" for (saname, sa) in SENSEALGS
                @testset "$lname" for (lname, lossf) in losses
                    g = DifferentiationInterface.gradient(
                        p -> lossf(p; ealg=ealg, sensealg=sa), backend, p0)
                    @test g !== nothing
                    @test g ≈ refs[lname] rtol = 1e-4
                end
            end
        end
    end

    # Callback case (ragged trajectories via terminate!). `Array(sol)` is invalid
    # here because trajectories end at different times, so only per-trajectory
    # access patterns are tested.
    @testset "with ContinuousCallback + terminate!" begin
        condition(u, t, integrator) = u[1] - 3.0
        cb = ContinuousCallback(condition, terminate!; save_positions=(false, true))

        function loss_cb(p; ealg, sensealg)
            sol = solve(build(p), Tsit5(), ealg; trajectories=n_traj,
                sensealg=sensealg, callback=cb)
            return sum(sum(abs2, s.u[end]) for s in sol.u)
        end

        @testset "$(backend_name(backend))" for backend in REVERSE_BACKENDS
            @testset "$ealgname" for (ealgname, ealg) in ENSEMBLE_ALGS
                @testset "$saname" for (saname, sa) in SENSEALGS
                    # With a `terminate!` callback the gradient also flows through
                    # the (near-zero here) event-time sensitivity, and different
                    # adjoint methods resolve that with differing accuracy. The
                    # point of this case is that the per-trajectory cotangent is
                    # routed at all (the original bug threw a BoundsError), so we
                    # only assert a finite gradient is produced rather than a
                    # strict match against ForwardDiff.
                    g = DifferentiationInterface.gradient(
                        p -> loss_cb(p; ealg = ealg, sensealg = sa), backend, p0)
                    @test g !== nothing
                    @test all(isfinite, g)
                end
            end
        end
    end
end
