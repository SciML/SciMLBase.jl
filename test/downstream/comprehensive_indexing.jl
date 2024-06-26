using ModelingToolkit, JumpProcesses, LinearAlgebra, NonlinearSolve, Optimization,
      OptimizationOptimJL, OrdinaryDiffEq, RecursiveArrayTools, SciMLBase,
      SteadyStateDiffEq, StochasticDiffEq, SymbolicIndexingInterface, Test
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
    if valp isa SciMLBase.NonlinearSolution && valp.prob isa SteadyStateProblem
        # Steady state problem indexing is broken, since the system is time-dependent but
        # the solution isn't
        @test_broken false
        continue
    end
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
    @info SciMLBase.parameterless_type(valp) typeof(indp)
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
