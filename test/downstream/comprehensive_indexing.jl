using ModelingToolkit, JumpProcesses, LinearAlgebra, NonlinearSolve, Optimization,
      OptimizationOptimJL, OrdinaryDiffEq, RecursiveArrayTools, SciMLBase,
      SteadyStateDiffEq, StochasticDiffEq, SymbolicIndexingInterface,
      DiffEqCallbacks, Test, Plots
using ModelingToolkit: t_nounits as t, D_nounits as D

# Sets rnd number.
using StableRNGs
rng = StableRNG(12345)
seed = rand(rng, 1:100)

### Basic Tests ###

# Prepares a model systems.
begin
    # Prepare system components.
    @parameters kp kd k1 k2
    @variables begin
        X(t), [bounds = (-10.0, 10.0)]
        Y(t), [bounds = (-10.0, 10.0)]
        XY(t)
    end
    alg_eqs = [0 ~ kp - kd * X - k1 * X + k2 * Y
               0 ~ 1 + k1 * X - k2 * Y - Y]
    diff_eqs = [D(X) ~ kp - kd * X - k1 * X + k2 * Y
                D(Y) ~ 1 + k1 * X - k2 * Y - Y]
    noise_eqs = [
        sqrt(kp + X),
        sqrt(k1 + Y)
    ]
    jumps = [
        ConstantRateJump(kp, [X ~ X + 1]),
        ConstantRateJump(kd * X, [X ~ X - 1]),
        ConstantRateJump(k1 * X, [X ~ X - 1, Y ~ Y + 1]),
        ConstantRateJump(k2 * Y, [X ~ X + 1, Y ~ Y - 1]),
        ConstantRateJump(1, [Y ~ Y + 1]),
        ConstantRateJump(Y, [Y ~ Y - 1])
    ]
    observed = [XY ~ X + Y]
    loss = kd * (k1 - X)^2 + k2 * (kp * Y - X^2)^2

    # Create systems (without structural_simplify, since that might modify systems to affect intended tests).
    osys = complete(ODESystem(diff_eqs, t; observed, name = :osys))
    ssys = complete(SDESystem(
        diff_eqs, noise_eqs, t, [X, Y], [kp, kd, k1, k2]; observed, name = :ssys))
    jsys = complete(JumpSystem(jumps, t, [X, Y], [kp, kd, k1, k2]; observed, name = :jsys))
    nsys = complete(NonlinearSystem(alg_eqs; observed, name = :nsys))
    optsys = complete(OptimizationSystem(
        loss, [X, Y], [kp, kd, k1, k2]; observed, name = :optsys))
end

# Prepares problems, integrators, and solutions.
begin
    # Sets problem inputs (to be used for all problem creations).
    u0_vals = [X => 4.0, Y => 5.0]
    tspan = (0.0, 10.0)
    p_vals = [kp => 1.0, kd => 0.1, k1 => 0.25, k2 => 0.5]

    # Creates problems.
    oprob = ODEProblem(osys, u0_vals, tspan, p_vals)
    sprob = SDEProblem(ssys, u0_vals, tspan, p_vals)
    dprob = DiscreteProblem(jsys, u0_vals, tspan, p_vals)
    jprob = JumpProblem(jsys, deepcopy(dprob), Direct(); rng)
    nprob = NonlinearProblem(nsys, u0_vals, p_vals)
    ssprob = SteadyStateProblem(osys, u0_vals, p_vals)
    optprob = OptimizationProblem(optsys, u0_vals, p_vals, grad = true, hess = true)
    problems = [oprob, sprob, dprob, jprob, nprob, ssprob, optprob]
    systems = [osys, ssys, jsys, jsys, nsys, osys, optsys]

    # Creates an `EnsembleProblem` for each problem.
    eoprob = EnsembleProblem(oprob)
    esprob = EnsembleProblem(sprob)
    edprob = EnsembleProblem(dprob)
    ejprob = EnsembleProblem(jprob)
    enprob = EnsembleProblem(nprob)
    essprob = EnsembleProblem(ssprob)
    eoptprob = EnsembleProblem(optprob)
    eproblems = [eoprob, esprob, edprob, ejprob, enprob, essprob, optprob]
    esystems = [osys, ssys, jsys, jsys, nsys, osys, optsys]

    # Creates integrators.
    oint = init(oprob, Tsit5(); save_everystep = false)
    sint = init(sprob, ImplicitEM(); save_everystep = false)
    jint = init(jprob, SSAStepper())
    nint = init(nprob, NewtonRaphson(); save_everystep = false)
    @test_broken ssint = init(ssprob, DynamicSS(Tsit5()); save_everystep = false) # https://github.com/SciML/SteadyStateDiffEq.jl/issues/79
    integrators = [oint, sint, jint, nint]
    integsystems = [osys, ssys, jsys, nsys]

    # Creates solutions.
    osol = solve(oprob, Tsit5())
    ssol = solve(sprob, ImplicitEM(); seed)
    jsol = solve(jprob, SSAStepper(); seed)
    nsol = solve(nprob, NewtonRaphson())
    sssol = solve(ssprob, DynamicSS(Tsit5()))
    optsol = solve(optprob, GradientDescent())
    sols = [osol, ssol, jsol, nsol, sssol, optsol]
end

non_timeseries_objects = [problems; eproblems; integrators; [nsol]; [sssol]; [optsol]]
non_timeseries_systems = [systems; esystems; integsystems; nsys; osys; optsys]
timeseries_objects = [osol, ssol, jsol]
timeseries_systems = [osys, ssys, jsys]

@testset "Non-timeseries indexing $(SciMLBase.parameterless_type(valp))" for (valp, indp) in zip(
    deepcopy(non_timeseries_objects), non_timeseries_systems)
    u = state_values(valp)
    uidxs = variable_index.((indp,), [X, Y])
    @testset "State indexing" begin
        for (sym, val, newval) in [(X, u[uidxs[1]], 4.0)
                                   (indp.X, u[uidxs[1]], 4.0)
                                   (:X, u[uidxs[1]], 4.0)
                                   (uidxs[1], u[uidxs[1]], 4.0)
                                   ([X, Y], u[uidxs], 4ones(2))
                                   ([indp.X, indp.Y], u[uidxs], 4ones(2))
                                   ([:X, :Y], u[uidxs], 4ones(2))
                                   (uidxs, u[uidxs], 4ones(2))
                                   ((X, Y), Tuple(u[uidxs]), (4.0, 4.0))
                                   ((indp.X, indp.Y), Tuple(u[uidxs]), (4.0, 4.0))
                                   ((:X, :Y), Tuple(u[uidxs]), (4.0, 4.0))
                                   (Tuple(uidxs), Tuple(u[uidxs]), (4.0, 4.0))]
            get = getu(indp, sym)
            set! = setu(indp, sym)
            @inferred get(valp)
            @test get(valp) == val
            if valp isa JumpProblem && sym isa Union{Tuple, AbstractArray}
                @test_broken valp[sym]
            else
                @test valp[sym] == val
            end

            if !(valp isa SciMLBase.AbstractNoTimeSolution)
                @inferred set!(valp, newval)
                @test get(valp) == newval
                set!(valp, val)
                @test get(valp) == val

                if !(valp isa JumpProblem) || !(sym isa Union{Tuple, AbstractArray})
                    valp[sym] = newval
                    @test valp[sym] == newval
                    valp[sym] = val
                    @test valp[sym] == val
                end
            end
        end
    end

    @testset "Observed" begin
        # Observed functions don't infer
        for (sym, val) in [(XY, sum(u))
                           (indp.XY, sum(u))
                           (:XY, sum(u))
                           ([X, indp.Y, :XY, X * Y], [u[uidxs]..., sum(u), prod(u)])
                           ((X, indp.Y, :XY, X * Y), (u[uidxs]..., sum(u), prod(u)))
                           (X * Y, prod(u))]
            get = getu(indp, sym)
            @test get(valp) == val
        end
    end

    getter = getu(indp, [])
    @test getter(valp) == []

    p = getindex.((Dict(p_vals),), [kp, kd, k1, k2])
    newp = p .* 10
    pidxs = parameter_index.((indp,), [kp, kd, k1, k2])
    @testset "Parameter indexing" begin
        for (sym, oldval, newval) in [(kp, p[1], newp[1])
                                      (indp.kp, p[1], newp[1])
                                      (:kp, p[1], newp[1])
                                      (pidxs[1], p[1], newp[1])
                                      ([kp, kd], p[1:2], newp[1:2])
                                      ([indp.kp, indp.kd], p[1:2], newp[1:2])
                                      ([:kp, :kd], p[1:2], newp[1:2])
                                      (pidxs[1:2], p[1:2], newp[1:2])
                                      ((kp, kd), Tuple(p[1:2]), Tuple(newp[1:2]))
                                      ((indp.kp, indp.kd), Tuple(p[1:2]), Tuple(newp[1:2]))
                                      ((:kp, :kd), Tuple(p[1:2]), Tuple(newp[1:2]))
                                      (Tuple(pidxs[1:2]), Tuple(p[1:2]), Tuple(newp[1:2]))]
            get = getp(indp, sym)
            set! = setp(indp, sym)

            @inferred get(valp)
            @test get(valp) == valp.ps[sym]
            @test get(valp) == oldval

            if !(valp isa SciMLBase.AbstractNoTimeSolution)
                @inferred set!(valp, newval)
                @test get(valp) == newval
                set!(valp, oldval)
                @test get(valp) == oldval

                valp.ps[sym] = newval
                @test get(valp) == newval
                valp.ps[sym] = oldval
                @test get(valp) == oldval
            end
        end
        getter = getp(indp, [])
        @test getter(valp) == []
    end
end

@testset "Timeseries indexing $(SciMLBase.parameterless_type(valp))" for (valp, indp) in zip(
    timeseries_objects, timeseries_systems)
    u = state_values(valp)
    uidxs = variable_index.((indp,), [X, Y])
    xvals = getindex.(valp.u, uidxs[1])
    yvals = getindex.(valp.u, uidxs[2])
    xyvals = xvals .+ yvals
    tvals = valp.t
    @testset "State indexing and observed" begin
        for (sym, val, check_inference, check_getindex) in [(X, xvals, true, true)
                                                            (indp.X, xvals, true, true)
                                                            (:X, xvals, true, true)
                                                            (uidxs[1], xvals, true, false)
                                                            ([X, Y], vcat.(xvals, yvals),
                                                                true, true)
                                                            ([indp.X, indp.Y],
                                                                vcat.(xvals, yvals),
                                                                true, true)
                                                            ([:X, :Y],
                                                                vcat.(xvals, yvals),
                                                                true, true)
                                                            (uidxs, vcat.(xvals, yvals),
                                                                true, false)
                                                            ((Y, X),
                                                                tuple.(yvals, xvals),
                                                                true, true)
                                                            ((indp.Y, indp.X),
                                                                tuple.(yvals, xvals),
                                                                true, true)
                                                            ((:Y, :X),
                                                                tuple.(yvals, xvals),
                                                                true, true)
                                                            (Tuple(reverse(uidxs)),
                                                                tuple.(yvals, xvals),
                                                                true, false)
                                                            (t, tvals, true, true)
                                                            (:t, tvals, true, true)
                                                            ([X, t], vcat.(xvals, tvals),
                                                                false, true)
                                                            ((Y, t),
                                                                tuple.(yvals, tvals),
                                                                true, true)
                                                            ([],
                                                                [[]
                                                                 for _ in 1:length(tvals)],
                                                                false,
                                                                false)
                                                            (XY, xyvals, true, true)
                                                            (indp.XY, xyvals, true, true)
                                                            (:XY, xyvals, true, true)
                                                            ([X, indp.Y, :XY, X * Y],
                                                                vcat.(xvals, yvals, xyvals,
                                                                    xvals .* yvals),
                                                                false,
                                                                true)
                                                            ((X, indp.Y, :XY, X * Y),
                                                                tuple.(
                                                                    xvals, yvals, xyvals,
                                                                    xvals .* yvals),
                                                                false,
                                                                true)
                                                            (X * Y, xvals .* yvals,
                                                                false, true)]
            get = getu(indp, sym)
            if check_inference
                @inferred get(valp)
            end
            @test get(valp) == val
            if check_getindex
                @test valp[sym] == val
            end
            # TODO: Test more subindexes when they're supported
            for i in [rand(eachindex(val)), CartesianIndex(1)]
                if check_inference
                    @inferred get(valp, i)
                end
                @test get(valp, i) == val[i]
                if check_getindex
                    @test valp[sym, i] == val[i]
                end
            end
        end
    end

    p = getindex.((Dict(p_vals),), [kp, kd, k1, k2])
    pidxs = parameter_index.((indp,), [kp, kd, k1, k2])

    @testset "Parameter indexing" begin
        for (sym, oldval) in [(kp, p[1])
                              (indp.kp, p[1])
                              (:kp, p[1])
                              (pidxs[1], p[1])
                              ([kp, kd], p[1:2])
                              ([indp.kp, indp.kd], p[1:2])
                              ([:kp, :kd], p[1:2])
                              (pidxs[1:2], p[1:2])
                              ((kp, kd), Tuple(p[1:2]))
                              ((indp.kp, indp.kd), Tuple(p[1:2]))
                              ((:kp, :kd), Tuple(p[1:2]))
                              (Tuple(pidxs[1:2]), Tuple(p[1:2]))]
            get = getp(indp, sym)

            @inferred get(valp)
            @test get(valp) == valp.ps[sym]
            @test get(valp) == oldval
        end
        getter = getp(indp, [])
        @test getter(valp) == []
    end

    @testset "Interpolation" begin
        sol = valp
        interpolated_sol = sol(0.0:1.0:10.0)
        @test interpolated_sol[XY] isa Vector
        @test interpolated_sol[XY, :] isa Vector
        @test interpolated_sol[XY, 2] isa Float64
        @test length(interpolated_sol[XY, 1:5]) == 5
        @test interpolated_sol[XY] ≈ interpolated_sol[X] .+ interpolated_sol[Y]
        @test collect(interpolated_sol[t]) isa Vector
        @test collect(interpolated_sol[t, :]) isa Vector
        @test interpolated_sol[t, 2] isa Float64
        @test length(interpolated_sol[t, 1:5]) == 5

        sol3 = sol(0.0:1.0:10.0, idxs = [X, Y])
        @test sol3.u isa Vector
        @test first(sol3.u) isa Vector
        @test length(sol3.u) == 11
        @test length(sol3.t) == 11
        @test collect(sol3[t]) ≈ sol3.t
        @test collect(sol3[t, 1:5]) ≈ sol3.t[1:5]
        @test sol(0.0:1.0:10.0, idxs = [Y, 1]) isa RecursiveArrayTools.DiffEqArray

        sol4 = sol(0.1, idxs = [X, Y])
        @test sol4 isa Vector
        @test length(sol4) == 2
        @test first(sol4) isa Real
        @test sol(0.1, idxs = [Y, 1]) isa Vector{<:Real}

        sol5 = sol(0.0:1.0:10.0, idxs = X)
        @test sol5.u isa Vector
        @test first(sol5.u) isa Real
        @test length(sol5.u) == 11
        @test length(sol5.t) == 11
        @test collect(sol5[t]) ≈ sol3.t
        @test collect(sol5[t, 1:5]) ≈ sol3.t[1:5]
        @test_throws Any sol(0.0:1.0:10.0, idxs = 1.2)

        sol6 = sol(0.1, idxs = X)
        @test sol6 isa Real
        @test_throws Any sol(0.1, idxs = 1.2)
    end
end

@testset "ODE with array symbolics" begin
    sts = @variables x(t)[1:3]=[1, 2, 3.0] y(t)=1.0
    ps = @parameters p[1:3] = [1, 2, 3]
    eqs = [collect(D.(x) .~ x)
           D(y) ~ norm(x) * y - x[1]]
    @named sys = ODESystem(eqs, t, [sts...;], ps)
    sys = complete(sys)
    prob = ODEProblem(sys, [], (0, 1.0))
    sol = solve(prob, Tsit5())
    # interpolation of array variables
    @test sol(1.0, idxs = x) == [sol(1.0, idxs = x[i]) for i in 1:3]

    x_idx = variable_index.((sys,), [x[1], x[2], x[3]])
    y_idx = variable_index(sys, y)
    x_val = getindex.(sol.u, (x_idx,))
    y_val = getindex.(sol.u, y_idx)
    obs_val = getindex.(x_val, 1) .+ y_val

    @testset "Solution indexing" begin
        # don't check inference for weird cases of nested arrays/tuples
        for (sym, val, check_inference) in [
            (x, x_val, true),
            (sys.x, x_val, true),
            (:x, x_val, true),
            (x_idx, x_val, true),
            (x[1] + sys.y, obs_val, true),
            ([x[1], x[2]], getindex.(x_val, ([1, 2],)), true),
            ([sys.x[1], sys.x[2]], getindex.(x_val, ([1, 2],)), true),
            ([x[1], x_idx[2]], getindex.(x_val, ([1, 2],)), false),
            ([x, x[1] + y], [[i, j] for (i, j) in zip(x_val, obs_val)], false),
            ([sys.x, x[1] + y], [[i, j] for (i, j) in zip(x_val, obs_val)], false),
            ([:x, x[1] + y], [[i, j] for (i, j) in zip(x_val, obs_val)], false),
            ([x, y], [[i, j] for (i, j) in zip(x_val, y_val)], false),
            ([sys.x, sys.y], [[i, j] for (i, j) in zip(x_val, y_val)], false),
            ([:x, :y], [[i, j] for (i, j) in zip(x_val, y_val)], false),
            ([x_idx, y_idx], [[i, j] for (i, j) in zip(x_val, y_val)], false),
            ([x, y_idx], [[i, j] for (i, j) in zip(x_val, y_val)], false),
            ([x, x], [[i, i] for i in x_val], true),
            ([sys.x, sys.x], [[i, i] for i in x_val], true),
            ([:x, :x], [[i, i] for i in x_val], true),
            ([x, x_idx], [[i, i] for i in x_val], false),
            ((x, y), [(i, j) for (i, j) in zip(x_val, y_val)], true),
            ((sys.x, sys.y), [(i, j) for (i, j) in zip(x_val, y_val)], true),
            ((:x, :y), [(i, j) for (i, j) in zip(x_val, y_val)], true),
            ((x, y_idx), [(i, j) for (i, j) in zip(x_val, y_val)], true),
            ((x, x), [(i, i) for i in x_val], true),
            ((sys.x, sys.x), [(i, i) for i in x_val], true),
            ((:x, :x), [(i, i) for i in x_val], true),
            ((x, x_idx), [(i, i) for i in x_val], true),
            ((x, x[1] + y), [(i, j) for (i, j) in zip(x_val, obs_val)], true),
            ((sys.x, x[1] + y), [(i, j) for (i, j) in zip(x_val, obs_val)], true),
            ((:x, x[1] + y), [(i, j) for (i, j) in zip(x_val, obs_val)], true),
            ((x, (x[1] + y, y)),
                [(i, (k, j)) for (i, j, k) in zip(x_val, y_val, obs_val)], false),
            ([x, [x[1] + y, y]],
                [[i, [k, j]] for (i, j, k) in zip(x_val, y_val, obs_val)], false),
            ((x, [x[1] + y, y], (x[1] + y, y_idx)),
                [(i, [k, j], (k, j)) for (i, j, k) in zip(x_val, y_val, obs_val)], false),
            ([x, [x[1] + y, y], (x[1] + y, y_idx)],
                [[i, [k, j], (k, j)] for (i, j, k) in zip(x_val, y_val, obs_val)], false)
        ]
            if check_inference
                @inferred getu(prob, sym)(sol)
            end
            @test getu(prob, sym)(sol) == val
        end
    end

    x_newval = [3.0, 6.0, 9.0]
    y_newval = 4.0
    x_probval = prob[x]
    y_probval = prob[y]

    @testset "Problem indexing" begin
        for (sym, oldval, newval, check_inference) in [
            (x, x_probval, x_newval, true),
            (sys.x, x_probval, x_newval, true),
            (:x, x_probval, x_newval, true),
            (x_idx, x_probval, x_newval, true),
            ((x, y), (x_probval, y_probval), (x_newval, y_newval), true),
            ((sys.x, sys.y), (x_probval, y_probval), (x_newval, y_newval), true),
            ((:x, :y), (x_probval, y_probval), (x_newval, y_newval), true),
            ((x_idx, y_idx), (x_probval, y_probval), (x_newval, y_newval), true),
            ([x, y], [x_probval, y_probval], [x_newval, y_newval], false),
            ([sys.x, sys.y], [x_probval, y_probval], [x_newval, y_newval], false),
            ([:x, :y], [x_probval, y_probval], [x_newval, y_newval], false),
            ([x_idx, y_idx], [x_probval, y_probval], [x_newval, y_newval], false),
            ((x, y_idx), (x_probval, y_probval), (x_newval, y_newval), true),
            ([x, y_idx], [x_probval, y_probval], [x_newval, y_newval], false),
            ((x_idx, y), (x_probval, y_probval), (x_newval, y_newval), true),
            ([x_idx, y], [x_probval, y_probval], [x_newval, y_newval], false),
            ([x[1:2], [y_idx, x[3]]], [x_probval[1:2], [y_probval, x_probval[3]]],
                [x_newval[1:2], [y_newval, x_newval[3]]], false),
            ([x[1:2], (y_idx, x[3])], [x_probval[1:2], (y_probval, x_probval[3])],
                [x_newval[1:2], (y_newval, x_newval[3])], false),
            ((x[1:2], [y_idx, x[3]]), (x_probval[1:2], [y_probval, x_probval[3]]),
                (x_newval[1:2], [y_newval, x_newval[3]]), false),
            ((x[1:2], (y_idx, x[3])), (x_probval[1:2], (y_probval, x_probval[3])),
                (x_newval[1:2], (y_newval, x_newval[3])), false)
        ]
            getter = getu(prob, sym)
            setter! = setu(prob, sym)
            if check_inference
                @inferred getter(prob)
            end
            @test getter(prob) == oldval
            if check_inference
                @inferred setter!(prob, newval)
            else
                setter!(prob, newval)
            end
            @test getter(prob) == newval
            setter!(prob, oldval)
            @test getter(prob) == oldval
        end
    end

    @testset "Parameter indexing" begin
        pval = [1.0, 2.0, 3.0]
        pval_new = [4.0, 5.0, 6.0]

        # don't check inference for nested tuples/arrays
        for (sym, oldval, newval, check_inference) in [
            (p[1], pval[1], pval_new[1], true),
            (p, pval, pval_new, true),
            (sys.p, pval, pval_new, true),
            (:p, pval, pval_new, true),
            ((p[1], p[2]), Tuple(pval[1:2]), Tuple(pval_new[1:2]), true),
            ([p[1], p[2]], pval[1:2], pval_new[1:2], true),
            ((p[1], p[2:3]), (pval[1], pval[2:3]), (pval_new[1], pval_new[2:3]), true),
            ([p[1], p[2:3]], [pval[1], pval[2:3]], [pval_new[1], pval_new[2:3]], false),
            ((p[1], (p[2],), [p[3]]), (pval[1], (pval[2],), [pval[3]]),
                (pval_new[1], (pval_new[2],), [pval_new[3]]), false),
            ([p[1], (p[2],), [p[3]]], [pval[1], (pval[2],), [pval[3]]],
                [pval_new[1], (pval_new[2],), [pval_new[3]]], false)
        ]
            getter = getp(prob, sym)
            setter! = setp(prob, sym)
            if check_inference
                @inferred getter(prob)
            end
            @test getter(prob) == oldval
            if check_inference
                @inferred setter!(prob, newval)
            else
                setter!(prob, newval)
            end
            @test getter(prob) == newval
            setter!(prob, oldval)
            @test getter(prob) == oldval
        end
    end
end

# Issue https://github.com/SciML/ModelingToolkit.jl/issues/2697
@testset "Interpolation of derivative of observed variables" begin
    @variables x(t) y(t) z(t) w(t)[1:2]
    @named sys = ODESystem(
        [D(x) ~ 1, y ~ x^2, z ~ 2y^2 + 3x, w[1] ~ x + y + z, w[2] ~ z * x * y], t)
    sys = structural_simplify(sys)
    prob = ODEProblem(sys, [x => 0.0], (0.0, 1.0))
    sol = solve(prob, Tsit5())
    @test_throws ErrorException sol(1.0, Val{1}, idxs = y)
    @test_throws ErrorException sol(1.0, Val{1}, idxs = [y, z])
    @test_throws ErrorException sol(1.0, Val{1}, idxs = w)
    @test_throws ErrorException sol(1.0, Val{1}, idxs = [w, w])
    @test_throws ErrorException sol(1.0, Val{1}, idxs = [w, y])
end

@testset "Discrete save indexing" begin
    struct NumSymbolCache{S}
        sc::S
    end
    SymbolicIndexingInterface.symbolic_container(s::NumSymbolCache) = s.sc
    function SymbolicIndexingInterface.is_observed(s::NumSymbolCache, x)
        return symbolic_type(x) != NotSymbolic() && !is_variable(s, x) &&
               !is_parameter(s, x) && !is_independent_variable(s, x)
    end
    function SymbolicIndexingInterface.observed(s::NumSymbolCache, x)
        res = ModelingToolkit.build_function(x,
            sort(variable_symbols(s); by = Base.Fix1(variable_index, s)),
            sort(parameter_symbols(s), by = Base.Fix1(parameter_index, s)),
            independent_variable_symbols(s)[]; expression = Val(false))
        if res isa Tuple
            return let oopfn = res[1], iipfn = res[2]
                fn(out, u, p, t) = iipfn(out, u, p, t)
                fn(u, p, t) = oopfn(u, p, t)
                fn
            end
        else
            return res
        end
    end
    function SymbolicIndexingInterface.parameter_observed(s::NumSymbolCache, x)
        if x isa Symbol
            allsyms = all_symbols(s)
            x = allsyms[findfirst(y -> hasname(y) && x == getname(y), allsyms)]
        elseif x isa AbstractArray
            allsyms = all_symbols(s)
            newx = []
            for i in eachindex(x)
                if x[i] isa Symbol
                    push!(newx, allsyms[findfirst(y -> hasname(y) && x[i] == getname(y), allsyms)])
                else
                    push!(newx, x[i])
                end
            end
            x = newx
        end
        res = ModelingToolkit.build_function(x,
            sort(parameter_symbols(s), by = Base.Fix1(parameter_index, s)),
            independent_variable_symbols(s)[]; expression = Val(false))
        if res isa Tuple
            return let oopfn = res[1], iipfn = res[2]
                fn(out, p, t) = iipfn(out, p, t)
                fn(p, t) = oopfn(p, t)
                fn
            end
        else
            return res
        end
    end
    function SymbolicIndexingInterface.get_all_timeseries_indexes(s::NumSymbolCache, x)
        if symbolic_type(x) == NotSymbolic()
            x = ModelingToolkit.unwrap.(x)
        else
            x = ModelingToolkit.unwrap(x)
        end
        if x isa Symbol
            allsyms = all_symbols(s)
            x = allsyms[findfirst(y -> hasname(y) && x == getname(y), allsyms)]
        elseif x isa AbstractArray
            allsyms = all_symbols(s)
            newx = []
            for i in eachindex(x)
                if x[i] isa Symbol
                    push!(newx, allsyms[findfirst(y -> hasname(y) && x[i] == getname(y), allsyms)])
                else
                    push!(newx, x[i])
                end
            end
            x = newx
        end
        vars = ModelingToolkit.vars(x)
        return mapreduce(union, vars; init = Set()) do sym
            if is_variable(s, sym)
                Set([ContinuousTimeseries()])
            elseif is_parameter(s, sym) && is_timeseries_parameter(s, sym)
                Set([timeseries_parameter_index(s, sym).timeseries_idx])
            else
                Set()
            end
        end
    end
    function SymbolicIndexingInterface.with_updated_parameter_timeseries_values(
            ::NumSymbolCache, p::Vector{Float64}, args...)
        for (idx, buf) in args
            if idx == 1
                p[1:2] .= buf
            else
                p[3:4] .= buf
            end
        end

        return p
    end
    function SciMLBase.create_parameter_timeseries_collection(s::NumSymbolCache, ps, tspan)
        trem = rem(tspan[1], 0.1, RoundDown)
        if trem > 0
            trem = 0.1 - trem
        end
        dea1 = DiffEqArray(Vector{Float64}[], (tspan[1] + trem):0.1:tspan[2])
        dea2 = DiffEqArray(Vector{Float64}[], Float64[])
        return ParameterTimeseriesCollection((dea1, dea2), deepcopy(ps))
    end
    function SciMLBase.get_saveable_values(::NumSymbolCache, p::Vector{Float64}, tsidx)
        if tsidx == 1
            return p[1:2]
        else
            return p[3:4]
        end
    end

    @variables x(t) ud1(t) ud2(t) xd1(t) xd2(t)
    @parameters kp
    sc = SymbolCache([x],
        Dict(ud1 => 1, xd1 => 2, ud2 => 3, xd2 => 4, kp => 5),
        t;
        timeseries_parameters = Dict(
            ud1 => ParameterTimeseriesIndex(1, 1), xd1 => ParameterTimeseriesIndex(1, 2),
            ud2 => ParameterTimeseriesIndex(2, 1), xd2 => ParameterTimeseriesIndex(2, 2)))
    sys = NumSymbolCache(sc)

    function f!(du, u, p, t)
        du .= u .* t .+ p[5] * sum(u)
    end
    fn = ODEFunction(f!; sys = sys)
    prob = ODEProblem(fn, [1.0], (0.0, 1.0), [1.0, 2.0, 3.0, 4.0, 5.0])
    cb1 = PeriodicCallback(0.1; initial_affect = true, final_affect = true,
        save_positions = (false, false)) do integ
        integ.p[1:2] .+= exp(-integ.t)
        SciMLBase.save_discretes!(integ, 1)
    end
    function affect2!(integ)
        integ.p[3:4] .+= only(integ.u)
        SciMLBase.save_discretes!(integ, 2)
    end
    cb2 = DiscreteCallback((args...) -> true, affect2!, save_positions = (false, false),
        initialize = (c, u, t, integ) -> affect2!(integ))
    sol = solve(deepcopy(prob), Tsit5(); callback = CallbackSet(cb1, cb2))

    ud1val = getindex.(sol.discretes.collection[1].u, 1)
    xd1val = getindex.(sol.discretes.collection[1].u, 2)
    ud2val = getindex.(sol.discretes.collection[2].u, 1)
    xd2val = getindex.(sol.discretes.collection[2].u, 2)

    for (sym, timeseries_index, val, buffer, isobs, check_inference) in [(ud1,
                                                                             1,
                                                                             ud1val,
                                                                             zeros(length(ud1val)),
                                                                             false,
                                                                             true)
                                                                         ([ud1, xd1],
                                                                             1,
                                                                             vcat.(ud1val,
                                                                                 xd1val),
                                                                             map(
                                                                                 _ -> zeros(2),
                                                                                 ud1val),
                                                                             false,
                                                                             true)
                                                                         ((ud2, xd2),
                                                                             2,
                                                                             tuple.(ud2val,
                                                                                 xd2val),
                                                                             map(
                                                                                 _ -> zeros(2),
                                                                                 ud2val),
                                                                             false,
                                                                             true)
                                                                         (ud2 + xd2,
                                                                             2,
                                                                             ud2val .+
                                                                             xd2val,
                                                                             zeros(length(ud2val)),
                                                                             true,
                                                                             true)
                                                                         (
                                                                             [ud2 + xd2,
                                                                                 ud2 * xd2],
                                                                             2,
                                                                             vcat.(
                                                                                 ud2val .+
                                                                                 xd2val,
                                                                                 ud2val .*
                                                                                 xd2val),
                                                                             map(
                                                                                 _ -> zeros(2),
                                                                                 ud2val),
                                                                             true,
                                                                             true)
                                                                         (
                                                                             (ud1 + xd1,
                                                                                 ud1 * xd1),
                                                                             1,
                                                                             tuple.(
                                                                                 ud1val .+
                                                                                 xd1val,
                                                                                 ud1val .*
                                                                                 xd1val),
                                                                             map(
                                                                                 _ -> zeros(2),
                                                                                 ud1val),
                                                                             true,
                                                                             true)]
        getter = getp(sys, sym)
        if check_inference
            @inferred getter(sol)
            @inferred getter(deepcopy(buffer), sol)
            if !isobs
                @inferred getter(parameter_values(sol))
                if !(eltype(val) <: Number)
                    @inferred getter(deepcopy(buffer[1]), parameter_values(sol))
                end
            end
        end

        @test getter(sol) == val
        if eltype(val) <: Number
            target = val
        else
            target = collect.(val)
        end
        tmp = deepcopy(buffer)
        getter(tmp, sol)
        @test tmp == target

        if !isobs
            @test getter(parameter_values(sol)) == val[end]
            if !(eltype(val) <: Number)
                target = collect(val[end])
                tmp = deepcopy(buffer)[end]
                getter(tmp, parameter_values(sol))
                @test tmp == target
            end
        end

        for subidx in [
            1, CartesianIndex(2), :, rand(Bool, length(val)), rand(eachindex(val), 4), 2:5]
            if check_inference
                @inferred getter(sol, subidx)
                if !isa(val[subidx], Number)
                    @inferred getter(deepcopy(buffer[subidx]), sol, subidx)
                end
            end
            @test getter(sol, subidx) == val[subidx]
            tmp = deepcopy(buffer[subidx])
            if val[subidx] isa Number
                continue
            end
            target = val[subidx]
            if eltype(target) <: Number
                target = collect(target)
            else
                target = collect.(target)
            end
            getter(tmp, sol, subidx)
            @test tmp == target
        end
    end

    for sym in [
        [ud1, xd1, ud2],
        (ud2, xd1, xd2),
        ud1 + ud2,
        [ud1 + ud2, ud1 * xd1],
        (ud1 + ud2, ud1 * xd1)]
        getter = getp(sys, sym)
        @test_throws Exception getter(sol)
        @test_throws Exception getter([], sol)
        for subidx in [1, CartesianIndex(1), :, rand(Bool, 4), rand(1:4, 3), 1:2]
            @test_throws Exception getter(sol, subidx)
            @test_throws Exception getter([], sol, subidx)
        end
    end

    kpval = sol.prob.p[5]
    xval = getindex.(sol.u)

    for (sym, val_is_timeseries, val, check_inference) in [
        (kp, false, kpval, true),
        ([kp, kp], false, [kpval, kpval], true),
        ((kp, kp), false, (kpval, kpval), true),
        (ud2, true, ud2val, true),
        ([ud2, kp], true, vcat.(ud2val, kpval), false),
        ((ud1, kp), true, tuple.(ud1val, kpval), false),
        ([kp, x], true, vcat.(kpval, xval), false),
        ((kp, x), true, tuple.(kpval, xval), false),
        (2ud2, true, 2 .* ud2val, true),
        ([kp, 2ud1], true, vcat.(kpval, 2 .* ud1val), false),
        ((kp, 2ud1), true, tuple.(kpval, 2 .* ud1val), false)
    ]
        getter = getu(sys, sym)
        if check_inference
            @inferred getter(sol)
        end
        @test getter(sol) == val
        reference = val_is_timeseries ? val : xval
        for subidx in [
            1, CartesianIndex(2), :, rand(Bool, length(reference)),
            rand(eachindex(reference), 4), 2:6
        ]
            if check_inference
                @inferred getter(sol, subidx)
            end
            target = if val_is_timeseries
                val[subidx]
            else
                val
            end
            @test getter(sol, subidx) == target
        end
    end

    _xval = xval[1]
    _ud1val = ud1val[1]
    _ud2val = ud2val[1]
    _xd1val = xd1val[1]
    _xd2val = xd2val[1]
    integ = init(prob, Tsit5(); callback = CallbackSet(cb1, cb2))
    for (sym, val, check_inference) in [
        ([x, ud1], [_xval, _ud1val], false),
        ((x, ud1), (_xval, _ud1val), true),
        (x + ud2, _xval + _ud2val, true),
        ([2x, 3xd1], [2_xval, 3_xd1val], true),
        ((2x, 3xd2), (2_xval, 3_xd2val), true)
    ]
        getter = getu(sys, sym)
        @test_throws Exception getter(sol)
        for subidx in [1, CartesianIndex(1), :, rand(Bool, 4), rand(1:4, 3), 1:2]
            @test_throws Exception getter(sol, subidx)
        end

        if check_inference
            @inferred getter(integ)
        end
        @test getter(integ) == val
    end

    xinterp = sol(0.1:0.1:0.3, idxs = x).u
    xinterp2 = sol(sol.discretes.collection[2].t[2:4], idxs = x).u
    ud1interp = ud1val[2:4]
    ud2interp = ud2val[2:4]

    c1 = SciMLBase.Clock(0.1)
    c2 = SciMLBase.SolverStepClock
    for (sym, t, val) in [
        (x, c1[2], xinterp[1]),
        (x, c1[2:4], xinterp),
        ([x, ud1], c1[2], [xinterp[1], ud1interp[1]]),
        ([x, ud1], c1[2:4], vcat.(xinterp, ud1interp)),
        (x, c2[2], xinterp2[1]),
        (x, c2[2:4], xinterp2),
        ([x, ud2], c2[2], [xinterp2[1], ud2interp[1]]),
        ([x, ud2], c2[2:4], vcat.(xinterp2, ud2interp))
    ]
        res = sol(t, idxs = sym)
        if res isa DiffEqArray
            res = res.u
        end
        @test res == val
    end

    @testset "Plotting" begin
        plotfn(t, u) = (t, 2u)
        all_idxs = [ud1, 2ud1, ud2, (plotfn, 0, ud1), (plotfn, t, ud1)]
        sym_idxs = [:ud1, :ud2, (plotfn, 0, :ud1), (plotfn, 0, :ud1)]
        
        for idx in Iterators.flatten((all_idxs, sym_idxs))
            @test_nowarn plot(sol; idxs = idx)
            @test_nowarn plot(sol; idxs = [idx])
        end
        for idx in Iterators.flatten((Iterators.product(all_idxs, all_idxs), Iterators.product(sym_idxs, sym_idxs)))
            @test_nowarn plot(sol; idxs = collect(idx))
            if !(idx[1] isa Tuple || idx[2] isa Tuple || length(get_all_timeseries_indexes(sol, collect(idx))) > 1)
                @test_nowarn plot(sol; idxs = idx)
            end
        end
    end
end
