using OrdinaryDiffEq, SciMLBase, SciMLSensitivity, Test
using ForwardDiff: ForwardDiff

# Zygote has compatibility issues on Julia 1.12+
if VERSION >= v"1.12"
    @info "Skipping ensemble adjoint tests on Julia 1.12+ due to AD compatibility issues"
    @testset "Ensemble adjoints (skipped on Julia 1.12+)" begin
        @test_skip false
    end
else
    using Zygote

    # Gradient correctness of AD through ensemble solves (SciML/SciMLSensitivity.jl#1478).
    # Two historical failure modes are covered:
    #   1. A `BoundsError` from `map`ing pullbacks over an `EnsembleSolution`/
    #      `VectorOfArray`-wrapped cotangent, which iterates scalars under
    #      RecursiveArrayTools v4.
    #   2. Silent gradient over-counting from Zygote's shared mutable-struct
    #      accumulator when the trajectory pullbacks read fields of the shared
    #      `ODEProblem`. Loss-decrease checks cannot catch this (the gradient is
    #      wrong only by trajectory-dependent positive factors), so these tests
    #      compare against ForwardDiff on per-trajectory reference solves.
    f_ens = (u, p, t) -> p .* u
    condition = (u, t, integrator) -> 0.5 - u[2]
    cb = ContinuousCallback(condition, terminate!)
    u0 = [1.0, 0.1]
    tspan = (0.0, 1.0)
    p0 = [1.0, 2.0]
    B = 3
    u0s = [Float64(i) .* u0 for i in 1:B]

    function ensemble_loss(p; ensalg, sensealg, callback = nothing, dense = false)
        prob = ODEProblem{false}(f_ens, u0, tspan, p)
        eprob = EnsembleProblem(
            prob;
            prob_func = (prob, ctx) -> remake(prob; u0 = u0s[ctx.sim_id]),
            safetycopy = false
        )
        kw = dense ? (; saveat = 0.1) : (;)
        sol = solve(
            eprob, Tsit5(), ensalg; trajectories = B, callback,
            abstol = 1.0e-10, reltol = 1.0e-10, sensealg, kw...
        )
        return if dense
            sum(abs2, Array(sol))
        else
            sum(abs2, reduce(hcat, [s.u[end] for s in sol.u]))
        end
    end

    function reference_loss(p; callback = nothing, dense = false)
        tot = zero(eltype(p))
        for i in 1:B
            prob = ODEProblem{false}(f_ens, u0s[i], tspan, p)
            kw = dense ? (; saveat = 0.1) : (;)
            s = solve(prob, Tsit5(); callback, abstol = 1.0e-10, reltol = 1.0e-10, kw...)
            tot += dense ? sum(abs2, Array(s)) : sum(abs2, s.u[end])
        end
        return tot
    end

    @testset "ensalg = $(nameof(typeof(ensalg))), sensealg = $(nameof(typeof(sensealg)))" for ensalg in (
                EnsembleSerial(), EnsembleThreads(),
            ),
            sensealg in (
                GaussAdjoint(autojacvec = ReverseDiffVJP()),
                QuadratureAdjoint(autojacvec = ReverseDiffVJP()),
                InterpolatingAdjoint(autojacvec = ReverseDiffVJP()),
            )

        # `dense = true` (a `saveat` grid + `Array(sol)`) is not combined with the
        # `terminate!` callback: early termination leaves the trajectories short of
        # the `saveat` grid, which is unsupported for discrete-cost adjoints
        # (SciMLSensitivity warns "Endpoints do not match").
        @testset "callback = $(callback !== nothing), dense = $dense" for (callback, dense) in (
                (nothing, false), (cb, false), (nothing, true),
            )

            gfd = ForwardDiff.gradient(p -> reference_loss(p; callback, dense), p0)
            g = Zygote.gradient(
                p -> ensemble_loss(p; ensalg, sensealg, callback, dense), p0
            )[1]
            @test g ≈ gfd rtol = 1.0e-4
        end
    end
end
