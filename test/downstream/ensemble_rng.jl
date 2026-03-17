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

# ============================================================
# Helpers
# ============================================================

# Extract endpoints from ensemble solution
endpoints(sim) = [sol[end] for sol in sim]

# Extract first positive time from each trajectory (jump time, skipping t=0 entries)
first_jump_times(sim) = [sol.t[findfirst(>(0), sol.t)] for sol in sim]

# Extract all solution time arrays (continuous-valued, safe for != comparisons)
all_timeseries(sim) = [sol.t for sol in sim]

# Build ensemble problem with default prob_func that perturbs u0.
# JumpProblems use identity prob_func — stochasticity comes from the jump process itself.
function make_eprob(prob; safetycopy = true)
    return if prob isa JumpProblem
        EnsembleProblem(prob; safetycopy)
    else
        EnsembleProblem(
            prob;
            prob_func = (prob, ctx) -> remake(prob; u0 = rand() * 0.1 .+ prob.u0),
            safetycopy,
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

# Jump problem/algorithm pairs for first-jump-time distinct stream tests.
# SSA only saves jump times by default; ODE+Jump and SDE+Jump need save_everystep=false
# to suppress adaptive timesteps and isolate jump times.
const SSA_PAIR = ("SSA", ssa_jprob, SSAStepper())
const CONTINUOUS_JUMP_PAIRS = [
    ("ODE+Jump", ode_jump_prob, Tsit5()),
    ("SDE+Jump", sde_jump_prob, SOSRI()),
]

# All ensemble algorithms for testing
const THREADED_ALGS = Threads.nthreads() >= 2 ?
    [("Serial", EnsembleSerial()), ("Threaded", EnsembleThreads())] :
    [("Serial", EnsembleSerial())]

# ============================================================
# 1. Seed reproducibility (Serial + Threaded, all problem types)
# ============================================================
@testset "1. Seed reproducibility" begin
    for (name, prob, alg) in SOLVER_PAIRS
        @testset "$name" begin
            eprob = make_eprob(prob)
            sim1 = solve(eprob, alg, EnsembleSerial(); seed = UInt64(42), trajectories = 8)
            sim2 = solve(eprob, alg, EnsembleSerial(); seed = UInt64(42), trajectories = 8)
            @test endpoints(sim1) == endpoints(sim2)

            if Threads.nthreads() >= 2
                sim3 = solve(
                    eprob, alg, EnsembleThreads();
                    seed = UInt64(42), trajectories = 8,
                )
                sim4 = solve(
                    eprob, alg, EnsembleThreads();
                    seed = UInt64(42), trajectories = 8,
                )
                @test endpoints(sim3) == endpoints(sim4)
            end
        end
    end
end

# ============================================================
# 2. rng kwarg reproducibility + priority over seed
# ============================================================
@testset "2. rng kwarg + priority" begin
    eprob = make_eprob(ode_prob)

    sim_rng1 = solve(
        eprob, Tsit5(), EnsembleSerial();
        rng = Random.Xoshiro(99), trajectories = 6,
    )
    sim_rng2 = solve(
        eprob, Tsit5(), EnsembleSerial();
        rng = Random.Xoshiro(99), trajectories = 6,
    )
    @test endpoints(sim_rng1) == endpoints(sim_rng2)

    # rng takes priority over conflicting seed
    sim_rng_seed = solve(
        eprob, Tsit5(), EnsembleSerial();
        rng = Random.Xoshiro(99), seed = UInt64(12345), trajectories = 6,
    )
    @test endpoints(sim_rng_seed) == endpoints(sim_rng1)
end

# ============================================================
# 3. Serial/Threaded/Distributed equivalence (all problem types)
# ============================================================
@testset "3. Cross-algorithm equivalence" begin
    for (name, prob, alg) in SOLVER_PAIRS
        @testset "$name" begin
            eprob = make_eprob(prob)
            sim_s = solve(
                eprob, alg, EnsembleSerial();
                seed = UInt64(77), trajectories = 8,
            )

            if Threads.nthreads() >= 2
                sim_t = solve(
                    eprob, alg, EnsembleThreads();
                    seed = UInt64(77), trajectories = 8,
                )
                @test endpoints(sim_s) == endpoints(sim_t)
            end

            sim_d = solve(
                eprob, alg, EnsembleDistributed();
                seed = UInt64(77), trajectories = 8,
            )
            @test endpoints(sim_s) == endpoints(sim_d)

            sim_st = solve(
                eprob, alg, EnsembleSplitThreads();
                seed = UInt64(77), trajectories = 8,
            )
            @test endpoints(sim_s) == endpoints(sim_st)
        end
    end
end

# ============================================================
# 4. Distinct streams — prob_func level (StableRNG end-to-end)
#    Verifies each trajectory's prob_func receives a distinct RNG stream.
#    Uses StableRNG for both master and rng_func → fully deterministic.
# ============================================================
@testset "4. Distinct streams (prob_func)" begin
    stable_rng_func = ctx -> StableRNG(ctx.sim_seed)
    eprob = EnsembleProblem(
        ode_prob;
        prob_func = (prob, ctx) ->
        remake(prob; u0 = rand(ctx.rng) * 0.1 + prob.u0),
    )

    for (alg_name, ensalg) in [
            THREADED_ALGS;
            ("Distributed", EnsembleDistributed());
            ("SplitThreads", EnsembleSplitThreads())
        ]
        @testset "$alg_name" begin
            sim = solve(
                eprob, Tsit5(), ensalg;
                rng = StableRNG(42), rng_func = stable_rng_func, trajectories = 20,
            )
            @test allunique(endpoints(sim))
        end
    end
end

# ============================================================
# 5. Distinct streams — solver level
#    Verifies each trajectory's solver uses a distinct RNG stream.
#    Jump problems: check allunique first positive jump times.
#    SDE problems: check allunique endpoints (continuous-valued).
# ============================================================
@testset "5. Distinct streams (solver)" begin
    # Jump problems: check first jump times are distinct.
    # SSA only saves jump times by default; ODE+Jump and SDE+Jump need
    # save_everystep=false to suppress ODE/SDE adaptive timesteps.
    @testset "Jump first-times" begin
        let (name, jprob, alg) = SSA_PAIR
            @testset "$name" begin
                eprob = EnsembleProblem(jprob)
                for (alg_name, ensalg) in [
                        THREADED_ALGS;
                        ("Distributed", EnsembleDistributed());
                        ("SplitThreads", EnsembleSplitThreads())
                    ]
                    @testset "$alg_name" begin
                        sim = solve(
                            eprob, alg, ensalg;
                            rng = StableRNG(42), trajectories = 20,
                        )
                        @test allunique(first_jump_times(sim))
                    end
                end
            end
        end

        for (name, jprob, alg) in CONTINUOUS_JUMP_PAIRS
            @testset "$name" begin
                eprob = EnsembleProblem(jprob)
                for (alg_name, ensalg) in [
                        THREADED_ALGS;
                        ("Distributed", EnsembleDistributed());
                        ("SplitThreads", EnsembleSplitThreads())
                    ]
                    @testset "$alg_name" begin
                        sim = solve(
                            eprob, alg, ensalg;
                            rng = StableRNG(42), trajectories = 20,
                            save_everystep = false,
                        )
                        @test allunique(first_jump_times(sim))
                    end
                end
            end
        end
    end

    # SDE: check endpoints are distinct (continuous-valued)
    @testset "SDE endpoints" begin
        eprob = EnsembleProblem(sde_prob)
        for (alg_name, ensalg) in [
                THREADED_ALGS;
                ("Distributed", EnsembleDistributed());
                ("SplitThreads", EnsembleSplitThreads())
            ]
            @testset "$alg_name" begin
                sim = solve(
                    eprob, SOSRI(), ensalg;
                    rng = StableRNG(42), trajectories = 20,
                )
                @test allunique(endpoints(sim))
            end
        end
    end
end

# ============================================================
# 5b. Distinct streams — explicit JumpProblem RNG
#     JumpProblem with explicit Xoshiro RNG (not TaskLocalRNG).
#     Without the seed fallback, deepcopy would give identical RNG states.
# ============================================================
@testset "5b. Distinct streams — explicit JumpProblem RNG" begin
    explicit_rng_jprob = JumpProblem(
        ssa_dprob, Direct(), birth_jump, death_jump;
        rng = Random.Xoshiro(999)
    )
    eprob = EnsembleProblem(explicit_rng_jprob)

    for (alg_name, ensalg) in [
            THREADED_ALGS;
            ("Distributed", EnsembleDistributed());
            ("SplitThreads", EnsembleSplitThreads())
        ]
        @testset "$alg_name" begin
            sim = solve(
                eprob, SSAStepper(), ensalg;
                rng = StableRNG(42), trajectories = 20
            )
            @test allunique(first_jump_times(sim))
        end
    end

    # Reproducibility: same seed → same results
    sim1 = solve(
        eprob, SSAStepper(), EnsembleSerial();
        seed = UInt64(42), trajectories = 10
    )
    sim2 = solve(
        eprob, SSAStepper(), EnsembleSerial();
        seed = UInt64(42), trajectories = 10
    )
    @test endpoints(sim1) == endpoints(sim2)

    # Different seeds → different results
    sim3 = solve(
        eprob, SSAStepper(), EnsembleSerial();
        seed = UInt64(123), trajectories = 10
    )
    @test first_jump_times(sim1) != first_jump_times(sim3)
end

# ============================================================
# 6. Different seeds → different results (all problem types)
# ============================================================
@testset "6. Different seeds → different results" begin
    for (name, prob, alg) in SOLVER_PAIRS
        @testset "$name" begin
            eprob = make_eprob(prob)
            sim_a = solve(
                eprob, alg, EnsembleSerial();
                seed = UInt64(42), trajectories = 6,
            )
            sim_b = solve(
                eprob, alg, EnsembleSerial();
                seed = UInt64(123), trajectories = 6,
            )
            @test all_timeseries(sim_a) != all_timeseries(sim_b)
        end
    end
end

# ============================================================
# 7. 5-arg prob_func reproducibility
# ============================================================
@testset "7. 2-arg prob_func with ctx.rng" begin
    eprob5 = EnsembleProblem(
        ode_prob;
        prob_func = (prob, ctx) ->
        remake(prob; u0 = rand(ctx.rng) * prob.u0),
    )
    sim1 = solve(eprob5, Tsit5(), EnsembleSerial(); seed = UInt64(55), trajectories = 6)
    sim2 = solve(eprob5, Tsit5(), EnsembleSerial(); seed = UInt64(55), trajectories = 6)
    @test endpoints(sim1) == endpoints(sim2)
end

# ============================================================
# 8. Custom rng_func (StableRNG)
# ============================================================
@testset "8. Custom rng_func (StableRNG)" begin
    custom_rng_func = ctx -> StableRNG(ctx.sim_seed)
    # Custom rng_func returns StableRNG (doesn't seed TaskLocalRNG),
    # so must use prob_func that explicitly uses ctx.rng.
    eprob_stable = EnsembleProblem(
        ode_prob;
        prob_func = (prob, ctx) ->
        remake(prob; u0 = rand(ctx.rng) * 0.1 + prob.u0),
    )

    sim1 = solve(
        eprob_stable, Tsit5(), EnsembleSerial();
        seed = UInt64(88), rng_func = custom_rng_func, trajectories = 6,
    )
    sim2 = solve(
        eprob_stable, Tsit5(), EnsembleSerial();
        seed = UInt64(88), rng_func = custom_rng_func, trajectories = 6,
    )
    @test endpoints(sim1) == endpoints(sim2)

    if Threads.nthreads() >= 2
        sim_t = solve(
            eprob_stable, Tsit5(), EnsembleThreads();
            seed = UInt64(88), rng_func = custom_rng_func, trajectories = 6,
        )
        @test endpoints(sim_t) == endpoints(sim1)
    end
end

# ============================================================
# 9. Batch size independence
# ============================================================
@testset "9. Batch size independence" begin
    eprob = make_eprob(ode_prob)
    sim_full = solve(
        eprob, Tsit5(), EnsembleSerial();
        seed = UInt64(33), trajectories = 12, batch_size = 12,
    )
    sim_batched = solve(
        eprob, Tsit5(), EnsembleSerial();
        seed = UInt64(33), trajectories = 12, batch_size = 5,
    )
    @test endpoints(sim_full) == endpoints(sim_batched)
end

# ============================================================
# 10. EnsembleContext field verification
# ============================================================
@testset "10. EnsembleContext fields" begin
    captured_ctxs = EnsembleContext[]
    eprob_ctx = EnsembleProblem(
        ode_prob;
        prob_func = (prob, ctx) -> begin
            push!(captured_ctxs, ctx)
            remake(prob; u0 = rand(ctx.rng) * prob.u0)
        end,
    )
    sim = solve(eprob_ctx, Tsit5(), EnsembleSerial(); seed = UInt64(42), trajectories = 5)

    @test length(captured_ctxs) == 5
    @test Set(ctx.sim_id for ctx in captured_ctxs) == Set(1:5)
    @test all(ctx.worker_id == 0 for ctx in captured_ctxs)
    @test all(ctx.sim_seed isa UInt64 for ctx in captured_ctxs)
    @test all(ctx.rng !== nothing for ctx in captured_ctxs)
    @test all(ctx.repeat == 1 for ctx in captured_ctxs)

    # Simulation seeds match the expected pre-generated values
    expected_seeds = SciMLBase.generate_sim_seeds(nothing, UInt64(42), 5)
    actual_seeds = [captured_ctxs[i].sim_seed for i in 1:5]
    @test Set(actual_seeds) == Set(expected_seeds)
end

# ============================================================
# 11. No seed graceful execution
# ============================================================
@testset "11. No seed graceful execution" begin
    eprob = EnsembleProblem(ode_prob)
    sim = solve(eprob, Tsit5(), EnsembleSerial(); trajectories = 4)
    @test length(sim) == 4
    @test all(sol.retcode == ReturnCode.Success for sol in sim)
end

# ============================================================
# 12. safetycopy=false isolation (task_local_storage path)
# ============================================================
@testset "12. safetycopy=false isolation" begin
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
                    seed = UInt64(42), trajectories = 50,
                )
                sim2 = solve(
                    eprob, alg, EnsembleThreads();
                    seed = UInt64(42), trajectories = 50,
                )
                @test endpoints(sim1) == endpoints(sim2)

                # Serial/Threaded equivalence
                sim_s = solve(
                    eprob, alg, EnsembleSerial();
                    seed = UInt64(42), trajectories = 50,
                )
                @test endpoints(sim_s) == endpoints(sim1)
            end

            # Distributed reproducibility
            sim_d1 = solve(
                eprob, alg, EnsembleDistributed();
                seed = UInt64(42), trajectories = 20,
            )
            sim_d2 = solve(
                eprob, alg, EnsembleDistributed();
                seed = UInt64(42), trajectories = 20,
            )
            @test endpoints(sim_d1) == endpoints(sim_d2)

            # Serial/Distributed equivalence
            sim_s = solve(
                eprob, alg, EnsembleSerial();
                seed = UInt64(42), trajectories = 20,
            )
            @test endpoints(sim_s) == endpoints(sim_d1)
        end
    end
end

# ============================================================
# 13. Threaded stress test (SSA, 400 trajectories)
# ============================================================
@testset "13. Threaded SSA stress test" begin
    if Threads.nthreads() >= 2
        eprob = make_eprob(ssa_jprob)
        sim1 = solve(
            eprob, SSAStepper(), EnsembleThreads();
            seed = UInt64(42), trajectories = 400,
        )
        sim2 = solve(
            eprob, SSAStepper(), EnsembleThreads();
            seed = UInt64(42), trajectories = 400,
        )
        @test endpoints(sim1) == endpoints(sim2)
    end
end

# ============================================================
# 14. Non-DE regression tests
# ============================================================
@testset "14. Non-DE regression" begin
    @testset "NonlinearSolve" begin
        f_nl(u, p) = u .* u .- p
        nlprob = NonlinearProblem(f_nl, [1.0, 1.0], 2.0)
        nl_eprob = EnsembleProblem(
            nlprob;
            prob_func = (prob, ctx) -> remake(prob; u0 = rand(2) .+ 0.5),
        )

        # Ensemble solve with seed succeeds (rng is NOT forwarded)
        sim1 = solve(
            nl_eprob, NewtonRaphson(), EnsembleSerial();
            seed = UInt64(42), trajectories = 5,
        )
        @test length(sim1) == 5

        # Reproducibility via TaskLocalRNG seeding
        sim2 = solve(
            nl_eprob, NewtonRaphson(), EnsembleSerial();
            seed = UInt64(42), trajectories = 5,
        )
        @test endpoints(sim1) == endpoints(sim2)
    end

    @testset "Optimization" begin
        rosenbrock(x, p) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
        optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
        optprob = OptimizationProblem(optf, zeros(2))
        opt_eprob = EnsembleProblem(
            optprob;
            prob_func = (prob, ctx) -> remake(prob; u0 = rand(2)),
        )

        # Ensemble solve with seed succeeds (rng is NOT forwarded)
        sim1 = Optimization.solve(
            opt_eprob, OptimizationOptimJL.BFGS(),
            EnsembleSerial(); seed = UInt64(42), trajectories = 4, maxiters = 10,
        )
        @test length(sim1) == 4

        # Reproducibility via TaskLocalRNG seeding
        sim2 = Optimization.solve(
            opt_eprob, OptimizationOptimJL.BFGS(),
            EnsembleSerial(); seed = UInt64(42), trajectories = 4, maxiters = 10,
        )
        @test endpoints(sim1) == endpoints(sim2)
    end
end

# ============================================================
# 15. Distributed smoke test (worker_id propagation)
# ============================================================
@testset "15. Distributed worker_id propagation" begin
    # worker_id is propagated (single-worker pmap runs on worker 1)
    dist_ctxs = EnsembleContext[]
    eprob_ctx = EnsembleProblem(
        ode_prob;
        prob_func = (prob, ctx) -> begin
            push!(dist_ctxs, ctx)
            prob
        end,
    )
    solve(
        eprob_ctx, Tsit5(), EnsembleDistributed();
        seed = UInt64(42), trajectories = 3,
    )
    @test length(dist_ctxs) == 3
    @test all(ctx.worker_id ∈ workers() for ctx in dist_ctxs)
end

# ============================================================
# 16. JumpProblem remake with parameter change — thread safety
# ============================================================
@testset "16. JumpProblem remake parameter change" begin
    # Small-population birth-death with low rates (fast simulation).
    # A callback at t_cb applies a large parameter-dependent jump to u[1]
    # and terminates. This tests that remake's aliased JumpProblem fields
    # don't cause race conditions on threads, and that the remade parameter
    # actually flows through to the solution.
    pbirth_rate(u, p, t) = p[1]
    pdeath_rate(u, p, t) = p[2] * u[1]
    pbirth_jump = ConstantRateJump(pbirth_rate, integrator -> (integrator.u[1] += 1))
    pdeath_jump = ConstantRateJump(pdeath_rate, integrator -> (integrator.u[1] -= 1))

    p0 = [0.1, 0.001, 0.0]
    pdprob = DiscreteProblem([5], (0.0, 100.0), p0)
    pjprob = JumpProblem(pdprob, Direct(), pbirth_jump, pdeath_jump)

    t_cb = 50.0
    pcb = DiscreteCallback(
        (u, t, integrator) -> t == t_cb,
        integrator -> begin
            integrator.u[1] += round(Int, integrator.p[3])
            reset_aggregated_jumps!(integrator)
            terminate!(integrator)
        end,
    )

    n_traj = 50
    pscales = [1000 * i for i in 1:n_traj]
    pf = (prob, ctx) ->
    remake(prob; p = [prob.prob.p[1], prob.prob.p[2], Float64(pscales[ctx.sim_id])])

    for safetycopy in (true, false)
        @testset "safetycopy=$safetycopy" begin
            eprob = EnsembleProblem(pjprob; prob_func = pf, safetycopy)

            sim = solve(
                eprob, SSAStepper(), EnsembleThreads();
                rng = StableRNG(42), trajectories = n_traj,
                callback = pcb, tstops = [t_cb],
            )

            # The callback saves pre/post states at t_cb then terminates.
            # post_cb - pre_cb must equal the trajectory's scale.
            for (idx, sol) in enumerate(sim)
                pre_cb = sol.u[end - 1][1]
                post_cb = sol.u[end][1]
                @test post_cb - pre_cb == pscales[idx]
            end

            # Reproducibility: same RNG → same results
            sim2 = solve(
                eprob, SSAStepper(), EnsembleThreads();
                rng = StableRNG(42), trajectories = n_traj,
                callback = pcb, tstops = [t_cb],
            )
            @test endpoints(sim) == endpoints(sim2)
        end
    end
end

# ============================================================
# 17. Backwards-compatible 5-arg solve_batch
# ============================================================
# DiffEqGPU calls SciMLBase.solve_batch with 5 positional args (no ensemble_rng_state).
# Verify the fallback methods work for Serial and Threaded.
@testset "17. Backwards-compatible 5-arg solve_batch" begin
    eprob = EnsembleProblem(
        ode_prob;
        prob_func = (prob, ctx) -> remake(prob; u0 = prob.u0 + 0.01 * ctx.sim_id),
    )

    result_serial = SciMLBase.solve_batch(
        eprob, Tsit5(), EnsembleSerial(), 1:5, nothing; saveat = [0.0, 0.5, 1.0]
    )
    @test length(result_serial) == 5
    @test all(sol.retcode == ReturnCode.Success for sol in result_serial)
    @test all(length(sol.t) == 3 for sol in result_serial)

    if Threads.nthreads() >= 2
        result_threaded = SciMLBase.solve_batch(
            eprob, Tsit5(), EnsembleThreads(), 1:5, nothing; saveat = [0.0, 0.5, 1.0]
        )
        @test length(result_threaded) == 5
        @test all(sol.retcode == ReturnCode.Success for sol in result_threaded)
        @test all(length(sol.t) == 3 for sol in result_threaded)
    end
end
