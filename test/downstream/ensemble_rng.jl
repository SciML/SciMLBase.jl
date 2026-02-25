using Test, Random, SciMLBase, Distributed
using OrdinaryDiffEq, StochasticDiffEq, JumpProcesses, StableRNGs
using NonlinearSolve, Optimization, OptimizationOptimJL, ForwardDiff

# ============================================================
# Problem definitions for each solver pathway
# ============================================================

# ODE
ode_f(u, p, t) = 1.01 * u
ode_prob = ODEProblem(ode_f, 0.5, (0.0, 1.0))

# SDE
sde_f(u, p, t) = 1.01 * u
sde_g(u, p, t) = 0.87 * u
sde_prob = SDEProblem(sde_f, sde_g, 0.5, (0.0, 1.0))

# SSA (pure jump)
birth_rate(u, p, t) = 10.0
death_rate(u, p, t) = 0.01 * u[1]
birth_affect!(integrator) = (integrator.u[1] += 1)
death_affect!(integrator) = (integrator.u[1] -= 1)
birth_jump = ConstantRateJump(birth_rate, birth_affect!)
death_jump = ConstantRateJump(death_rate, death_affect!)
ssa_dprob = DiscreteProblem([100], (0.0, 10.0))
ssa_jprob = JumpProblem(ssa_dprob, Direct(), birth_jump, death_jump)

# ODE+Jump
ode_jump_prob = JumpProblem(
    ODEProblem((du, u, p, t) -> (du .= 0.0), [100.0], (0.0, 10.0)),
    Direct(), birth_jump, death_jump,
)

# SDE+Jump
sde_jump_prob = JumpProblem(
    SDEProblem(
        (du, u, p, t) -> (du .= 0.01 .* u),
        (du, u, p, t) -> (du .= 0.1 .* u),
        [100.0], (0.0, 10.0),
    ),
    Direct(), birth_jump, death_jump,
)

# Helper: extract endpoints from ensemble solution
endpoints(sim) = [sol[end] for sol in sim]

# Helper: build ensemble problem with default prob_func using rand()
# JumpProblem doesn't expose u0 directly, so use identity prob_func for those —
# stochasticity comes from the jump process itself.
function make_eprob(prob; safetycopy = true)
    return if prob isa JumpProblem
        EnsembleProblem(prob; safetycopy)
    else
        EnsembleProblem(
            prob;
            prob_func = (prob, i, repeat) -> remake(prob; u0 = rand() * 0.1 .+ prob.u0),
            safetycopy
        )
    end
end

# Problem/algorithm pairs — JumpProblems need explicit algorithms
const SOLVER_PAIRS = [
    ("ODE", ode_prob, Tsit5()),
    ("SDE", sde_prob, SOSRI()),
    ("SSA", ssa_jprob, SSAStepper()),
    ("ODE+Jump", ode_jump_prob, Tsit5()),
    ("SDE+Jump", sde_jump_prob, SOSRI()),
]

# ============================================================
# A. Seed reproducibility (Serial + Threaded)
# ============================================================
@testset "A. Seed reproducibility" begin
    for (name, prob, alg) in SOLVER_PAIRS
        @testset "$name" begin
            eprob = make_eprob(prob)
            sim1 = solve(eprob, alg, EnsembleSerial(); seed = UInt64(42), trajectories = 8)
            sim2 = solve(eprob, alg, EnsembleSerial(); seed = UInt64(42), trajectories = 8)
            @test endpoints(sim1) == endpoints(sim2)

            if Threads.nthreads() >= 2
                sim3 = solve(eprob, alg, EnsembleThreads(); seed = UInt64(42), trajectories = 8)
                sim4 = solve(eprob, alg, EnsembleThreads(); seed = UInt64(42), trajectories = 8)
                @test endpoints(sim3) == endpoints(sim4)
            end
        end
    end
end

# ============================================================
# B. rng kwarg reproducibility + priority over seed
# ============================================================
@testset "B. rng kwarg + priority" begin
    eprob = make_eprob(ode_prob)

    sim_rng1 = solve(
        eprob, Tsit5(), EnsembleSerial();
        rng = Random.Xoshiro(99), trajectories = 6
    )
    sim_rng2 = solve(
        eprob, Tsit5(), EnsembleSerial();
        rng = Random.Xoshiro(99), trajectories = 6
    )
    @test endpoints(sim_rng1) == endpoints(sim_rng2)

    # rng takes priority over conflicting seed
    sim_rng_seed = solve(
        eprob, Tsit5(), EnsembleSerial();
        rng = Random.Xoshiro(99), seed = UInt64(12345), trajectories = 6
    )
    @test endpoints(sim_rng_seed) == endpoints(sim_rng1)
end

# ============================================================
# C. Serial/Threaded equivalence
# ============================================================
@testset "C. Serial/Threaded equivalence" begin
    if Threads.nthreads() >= 2
        for (name, prob, alg) in [
                ("ODE", ode_prob, Tsit5()),
                ("SDE", sde_prob, SOSRI()),
                ("SSA", ssa_jprob, SSAStepper()),
                ("ODE+Jump", ode_jump_prob, Tsit5()),
                ("SDE+Jump", sde_jump_prob, SOSRI()),
            ]
            @testset "$name" begin
                eprob = make_eprob(prob)
                sim_s = solve(
                    eprob, alg, EnsembleSerial();
                    seed = UInt64(77), trajectories = 8
                )
                sim_t = solve(
                    eprob, alg, EnsembleThreads();
                    seed = UInt64(77), trajectories = 8
                )
                @test endpoints(sim_s) == endpoints(sim_t)
            end
        end
    end
end

# ============================================================
# E. 5-arg prob_func reproducibility
# ============================================================
@testset "E. 5-arg prob_func" begin
    eprob5 = EnsembleProblem(
        ode_prob;
        prob_func = (prob, i, repeat, rng, ctx) ->
        remake(prob; u0 = rand(rng) * prob.u0)
    )
    sim1 = solve(eprob5, Tsit5(), EnsembleSerial(); seed = UInt64(55), trajectories = 6)
    sim2 = solve(eprob5, Tsit5(), EnsembleSerial(); seed = UInt64(55), trajectories = 6)
    @test endpoints(sim1) == endpoints(sim2)
end

# ============================================================
# F. Custom rng_func (StableRNG)
# ============================================================
@testset "F. Custom rng_func (StableRNG)" begin
    custom_rng_func = ctx -> StableRNG(ctx.trajectory_seed)
    # Custom rng_func returns StableRNG (doesn't seed TaskLocalRNG),
    # so must use 5-arg prob_func that explicitly uses the provided rng.
    eprob_stable = EnsembleProblem(
        ode_prob;
        prob_func = (prob, i, repeat, rng, ctx) ->
        remake(prob; u0 = rand(rng) * 0.1 + prob.u0)
    )

    sim1 = solve(
        eprob_stable, Tsit5(), EnsembleSerial();
        seed = UInt64(88), rng_func = custom_rng_func, trajectories = 6
    )
    sim2 = solve(
        eprob_stable, Tsit5(), EnsembleSerial();
        seed = UInt64(88), rng_func = custom_rng_func, trajectories = 6
    )
    @test endpoints(sim1) == endpoints(sim2)

    if Threads.nthreads() >= 2
        sim_t = solve(
            eprob_stable, Tsit5(), EnsembleThreads();
            seed = UInt64(88), rng_func = custom_rng_func, trajectories = 6
        )
        @test endpoints(sim_t) == endpoints(sim1)
    end
end

# ============================================================
# G. Batch size independence
# ============================================================
@testset "G. Batch size independence" begin
    eprob = make_eprob(ode_prob)
    sim_full = solve(
        eprob, Tsit5(), EnsembleSerial();
        seed = UInt64(33), trajectories = 12, batch_size = 12
    )
    sim_batched = solve(
        eprob, Tsit5(), EnsembleSerial();
        seed = UInt64(33), trajectories = 12, batch_size = 5
    )
    @test endpoints(sim_full) == endpoints(sim_batched)
end

# ============================================================
# H. EnsembleContext field verification
# ============================================================
@testset "H. EnsembleContext fields" begin
    captured_ctxs = EnsembleContext[]
    eprob_ctx = EnsembleProblem(
        ode_prob;
        prob_func = (prob, i, repeat, rng, ctx) -> begin
            push!(captured_ctxs, ctx)
            remake(prob; u0 = rand(rng) * prob.u0)
        end
    )
    sim = solve(eprob_ctx, Tsit5(), EnsembleSerial(); seed = UInt64(42), trajectories = 5)

    @test length(captured_ctxs) == 5
    @test Set(ctx.global_trajectory_id for ctx in captured_ctxs) == Set(1:5)
    @test all(ctx.worker_id == 0 for ctx in captured_ctxs)
    @test all(ctx.trajectory_seed isa UInt64 for ctx in captured_ctxs)

    # Trajectory seeds match the expected pre-generated values
    expected_seeds = SciMLBase.generate_trajectory_seeds(nothing, UInt64(42), 5)
    actual_seeds = [captured_ctxs[i].trajectory_seed for i in 1:5]
    @test Set(actual_seeds) == Set(expected_seeds)
end

# ============================================================
# I. No seed graceful execution
# ============================================================
@testset "I. No seed graceful execution" begin
    eprob = EnsembleProblem(ode_prob)
    sim = solve(eprob, Tsit5(), EnsembleSerial(); trajectories = 4)
    @test length(sim) == 4
    @test all(sol.retcode == ReturnCode.Success for sol in sim)
end

# ============================================================
# J. Different seeds produce different results
# ============================================================
@testset "J. Different seeds → different results" begin
    eprob = make_eprob(ode_prob)
    sim_a = solve(eprob, Tsit5(), EnsembleSerial(); seed = UInt64(42), trajectories = 6)
    sim_b = solve(eprob, Tsit5(), EnsembleSerial(); seed = UInt64(123), trajectories = 6)
    @test endpoints(sim_a) != endpoints(sim_b)
end

# ============================================================
# K. Threaded stress test (SSA, 400 trajectories)
# ============================================================
@testset "K. Threaded SSA stress test" begin
    if Threads.nthreads() >= 2
        eprob = make_eprob(ssa_jprob)
        sim1 = solve(eprob, SSAStepper(), EnsembleThreads(); seed = UInt64(42), trajectories = 400)
        sim2 = solve(eprob, SSAStepper(), EnsembleThreads(); seed = UInt64(42), trajectories = 400)
        @test endpoints(sim1) == endpoints(sim2)
    end
end

# ============================================================
# K2. JumpProblem safetycopy=false isolation (task_local_storage path)
# ============================================================
@testset "K2. safetycopy=false isolation" begin
    for (name, jprob, alg) in [
            ("SDE", sde_prob, SOSRI()),
            ("SSA", ssa_jprob, SSAStepper()),
            ("ODE+Jump", ode_jump_prob, Tsit5()),
            ("SDE+Jump", sde_jump_prob, SOSRI()),
        ]
        @testset "$name" begin
            eprob = make_eprob(jprob; safetycopy = false)

            # Threaded reproducibility
            if Threads.nthreads() >= 2
                sim1 = solve(
                    eprob, alg, EnsembleThreads();
                    seed = UInt64(42), trajectories = 50
                )
                sim2 = solve(
                    eprob, alg, EnsembleThreads();
                    seed = UInt64(42), trajectories = 50
                )
                @test endpoints(sim1) == endpoints(sim2)

                # Serial/Threaded equivalence
                sim_s = solve(
                    eprob, alg, EnsembleSerial();
                    seed = UInt64(42), trajectories = 50
                )
                @test endpoints(sim_s) == endpoints(sim1)
            end

            # Distributed reproducibility
            sim_d1 = solve(
                eprob, alg, EnsembleDistributed();
                seed = UInt64(42), trajectories = 20
            )
            sim_d2 = solve(
                eprob, alg, EnsembleDistributed();
                seed = UInt64(42), trajectories = 20
            )
            @test endpoints(sim_d1) == endpoints(sim_d2)

            # Serial/Distributed equivalence
            sim_s = solve(
                eprob, alg, EnsembleSerial();
                seed = UInt64(42), trajectories = 20
            )
            @test endpoints(sim_s) == endpoints(sim_d1)
        end
    end
end

# ============================================================
# L. Non-DE regression tests
# ============================================================
@testset "L. Non-DE regression" begin
    @testset "NonlinearSolve" begin
        f_nl(u, p) = u .* u .- p
        nlprob = NonlinearProblem(f_nl, [1.0, 1.0], 2.0)
        nl_eprob = EnsembleProblem(
            nlprob;
            prob_func = (prob, i, repeat) -> remake(prob; u0 = rand(2) .+ 0.5)
        )

        # Ensemble solve with seed succeeds (rng is NOT forwarded)
        sim1 = solve(
            nl_eprob, NewtonRaphson(), EnsembleSerial();
            seed = UInt64(42), trajectories = 5
        )
        @test length(sim1) == 5

        # Reproducibility via TaskLocalRNG seeding
        sim2 = solve(
            nl_eprob, NewtonRaphson(), EnsembleSerial();
            seed = UInt64(42), trajectories = 5
        )
        @test endpoints(sim1) == endpoints(sim2)
    end

    @testset "Optimization" begin
        rosenbrock(x, p) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
        optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
        optprob = OptimizationProblem(optf, zeros(2))
        opt_eprob = EnsembleProblem(
            optprob;
            prob_func = (prob, i, repeat) -> remake(prob; u0 = rand(2))
        )

        # Ensemble solve with seed succeeds (rng is NOT forwarded)
        sim1 = Optimization.solve(
            opt_eprob, OptimizationOptimJL.BFGS(),
            EnsembleSerial(); seed = UInt64(42), trajectories = 4, maxiters = 10
        )
        @test length(sim1) == 4

        # Reproducibility via TaskLocalRNG seeding
        sim2 = Optimization.solve(
            opt_eprob, OptimizationOptimJL.BFGS(),
            EnsembleSerial(); seed = UInt64(42), trajectories = 4, maxiters = 10
        )
        @test endpoints(sim1) == endpoints(sim2)
    end
end

# ============================================================
# M. Distributed smoke test (single-worker pmap, ODE only)
# ============================================================
@testset "M. Distributed smoke test" begin
    eprob = make_eprob(ode_prob)

    # EnsembleDistributed with seed completes
    sim1 = solve(eprob, Tsit5(), EnsembleDistributed(); seed = UInt64(42), trajectories = 6)
    @test length(sim1) == 6

    # Reproducibility
    sim2 = solve(eprob, Tsit5(), EnsembleDistributed(); seed = UInt64(42), trajectories = 6)
    @test endpoints(sim1) == endpoints(sim2)

    # Serial/Distributed equivalence
    sim_s = solve(eprob, Tsit5(), EnsembleSerial(); seed = UInt64(42), trajectories = 6)
    @test endpoints(sim_s) == endpoints(sim1)

    # worker_id is propagated (single-worker pmap runs on worker 1)
    dist_ctxs = EnsembleContext[]
    eprob_ctx = EnsembleProblem(
        ode_prob;
        prob_func = (prob, i, repeat, rng, ctx) -> begin
            push!(dist_ctxs, ctx)
            prob
        end
    )
    solve(
        eprob_ctx, Tsit5(), EnsembleDistributed();
        seed = UInt64(42), trajectories = 3
    )
    @test length(dist_ctxs) == 3
    @test all(ctx.worker_id ∈ workers() for ctx in dist_ctxs)
end
