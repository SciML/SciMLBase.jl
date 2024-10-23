using ModelingToolkit, OrdinaryDiffEq, RecursiveArrayTools, StochasticDiffEq, Test
using StochasticDiffEq
using SymbolicIndexingInterface
using ModelingToolkit: t_nounits as t, D_nounits as D
using Plots: Plots, plot

### Tests on non-layered model (everything should work). ###

@parameters a b c d
@variables s1(t) s2(t)

eqs = [D(s1) ~ a * s1 / (1 + s1 + s2) - b * s1,
    D(s2) ~ +c * s2 / (1 + s1 + s2) - d * s2]

@mtkbuild population_model = ODESystem(eqs, t)

# Tests on ODEProblem.
u0 = [s1 => 2.0, s2 => 1.0]
p = [a => 2.0, b => 1.0, c => 1.0, d => 1.0]
tspan = (0.0, 1000000.0)
oprob = ODEProblem(population_model, u0, tspan, p)
sol = solve(oprob, Rodas4())

@test sol[s1] == sol[population_model.s1] == sol[:s1]
@test sol[s2] == sol[population_model.s2] == sol[:s2]
@test sol[s1][end] ≈ 1.0
@test_throws Exception sol[a]
@test_throws Exception sol[population_model.a]
@test_throws Exception sol[:a]

@testset "plot ODE solution" begin
    Plots.unicodeplots()
    f = ODEFunction((u, p, t) -> -u, analytic = (u0, p, t) -> u0 * exp(-t))

    # scalar
    ode = ODEProblem(f, 1.0, (0.0, 1.0))
    sol = solve(ode, Tsit5())
    @test_nowarn plot(sol)
    @test_nowarn plot(sol; plot_analytic = true)

    # vector
    ode = ODEProblem(f, [1.0, 2.0], (0.0, 1.0))
    sol = solve(ode, Tsit5())
    @test_nowarn plot(sol)
    @test_nowarn plot(sol; plot_analytic = true)

    # matrix
    ode = ODEProblem(f, [1.0 2.0; 3.0 4.0], (0.0, 1.0))
    sol = solve(ode, Tsit5())
    @test_nowarn plot(sol)
    @test_nowarn plot(sol; plot_analytic = true)
end

# Tests on SDEProblem
noiseeqs = [0.1 * s1,
    0.1 * s2]
@named noisy_population_model = SDESystem(population_model, noiseeqs)
noisy_population_model = complete(noisy_population_model)
sprob = SDEProblem(noisy_population_model, u0, (0.0, 100.0), p)
sol = solve(sprob, ImplicitEM())

@test sol[s1] == sol[noisy_population_model.s1] == sol[:s1]
@test sol[s2] == sol[noisy_population_model.s2] == sol[:s2]
@test_throws Exception sol[a]
@test_throws Exception sol[noisy_population_model.a]
@test_throws Exception sol[:a]
@test_nowarn sol(0.5, idxs = noisy_population_model.s1)
### Tests on layered model (some things should not work). ###

@parameters σ ρ β
@variables x(t) y(t) z(t)

eqs = [D(x) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z]

@named lorenz1 = ODESystem(eqs, t)
@named lorenz2 = ODESystem(eqs, t)

@parameters γ
@variables a(t) α(t)
connections = [0 ~ lorenz1.x + lorenz2.y + a * γ,
    α ~ 2lorenz1.x + a * γ]
@mtkbuild sys = ODESystem(connections, t, [a, α], [γ], systems = [lorenz1, lorenz2])

u0 = [lorenz1.x => 1.0,
    lorenz1.y => 0.0,
    lorenz1.z => 0.0,
    lorenz2.x => 0.0,
    lorenz2.y => 1.0,
    lorenz2.z => 0.0]

p = [lorenz1.σ => 10.0,
    lorenz1.ρ => 28.0,
    lorenz1.β => 8 / 3,
    lorenz2.σ => 10.0,
    lorenz2.ρ => 28.0,
    lorenz2.β => 8 / 3,
    γ => 2.0]

tspan = (0.0, 100.0)
prob = ODEProblem(sys, u0, tspan, p)
sol = solve(prob, Rodas4())

@test_throws ArgumentError sol[x]
@test in(sol[lorenz1.x], [getindex.(sol.u, 1) for i in 1:length(unknowns(sol.prob.f.sys))])
@test_throws KeyError sol[:x]

### Non-symbolic indexing tests
@test sol[:, 1] isa AbstractVector
@test sol[:, 1:2] isa AbstractDiffEqArray
@test sol[:, [1, 2]] isa AbstractDiffEqArray

sol1 = sol(0.0:1.0:10.0)
@test sol1.u isa Vector
@test first(sol1.u) isa Vector
@test length(sol1.u) == 11
@test length(sol1.t) == 11

sol2 = sol(0.1)
@test sol2 isa Vector
@test length(sol2) == length(unknowns(sys))
@test first(sol2) isa Real

sol3 = sol(0.0:1.0:10.0, idxs = [lorenz1.x, lorenz2.x])

sol7 = sol(0.0:1.0:10.0, idxs = [2, 1])
@test sol7.u isa Vector
@test first(sol7.u) isa Vector
@test length(sol7.u) == 11
@test length(sol7.t) == 11
@test collect(sol7[t]) ≈ sol3.t
@test collect(sol7[t, 1:5]) ≈ sol3.t[1:5]

sol8 = sol(0.1, idxs = [2, 1])
@test sol8 isa Vector
@test length(sol8) == 2
@test first(sol8) isa Real

sol9 = sol(0.0:1.0:10.0, idxs = 2)
@test sol9.u isa Vector
@test first(sol9.u) isa Real
@test length(sol9.u) == 11
@test length(sol9.t) == 11
@test collect(sol9[t]) ≈ sol3.t
@test collect(sol9[t, 1:5]) ≈ sol3.t[1:5]

sol10 = sol(0.1, idxs = 2)
@test sol10 isa Real

@testset "Plot idxs" begin
    @variables x(t) y(t)
    @parameters p
    @mtkbuild sys = ODESystem([D(x) ~ x * t, D(y) ~ y - p * x], t)
    prob = ODEProblem(sys, [x => 1.0, y => 2.0], (0.0, 1.0), [p => 1.0])
    sol = solve(prob, Tsit5())

    plotfn(t, u) = (t, 2u)
    all_idxs = [x, x + p * y, t, (plotfn, 0, 1), (plotfn, t, 1), (plotfn, 0, x),
        (plotfn, t, x), (plotfn, t, p * y)]
    sym_idxs = [:x, :t, (plotfn, :t, 1), (plotfn, 0, :x),
        (plotfn, :t, :x)]
    for idx in Iterators.flatten((all_idxs, sym_idxs))
        @test_nowarn plot(sol; idxs = idx)
        @test_nowarn plot(sol; idxs = [idx])
    end
    for idx in Iterators.flatten((
        Iterators.product(all_idxs, all_idxs), Iterators.product(sym_idxs, sym_idxs)))
        @test_nowarn plot(sol; idxs = collect(idx))
        if !(idx[1] isa Tuple || idx[2] isa Tuple)
            @test_nowarn plot(sol; idxs = idx)
        end
    end
end

@testset "Saved subsystem" begin
    @testset "Purely continuous ODE/DAE/SDE-solutions" begin
        @variables x(t) y(t)
        @parameters p
        @mtkbuild sys = ODESystem([D(x) ~ x + p * y, D(y) ~ 2p + x^2], t)
        @test length(unknowns(sys)) == 2
        xidx = variable_index(sys, x)
        prob = ODEProblem(sys, [x => 1.0, y => 1.0], (0.0, 5.0), [p => 0.5])

        @test SciMLBase.SavedSubsystem(sys, prob.p, []) ===
              SciMLBase.SavedSubsystem(sys, prob.p, nothing) === nothing
        @test SciMLBase.SavedSubsystem(sys, prob.p, [x, y]) === nothing
        @test begin
            ss1 = SciMLBase.SavedSubsystem(sys, prob.p, [x])
            ss2 = SciMLBase.SavedSubsystem(sys, prob.p, [xidx])
            ss1.state_map == ss2.state_map
        end

        ode_sol = solve(prob, Tsit5(); save_idxs = xidx)
        subsys = SciMLBase.SavedSubsystem(sys, prob.p, [xidx])
        # FIXME: hack for save_idxs
        SciMLBase.@reset ode_sol.saved_subsystem = subsys

        @mtkbuild sys = ODESystem([D(x) ~ x + p * y, 1 ~ sin(y) + cos(x)], t)
        xidx = variable_index(sys, x)
        prob = DAEProblem(sys, [D(x) => x + p * y, D(y) => 1 / sqrt(1 - (1 - cos(x))^2)],
            [x => 1.0, y => asin(1 - cos(x))], (0.0, 1.0), [p => 2.0])
        dae_sol = solve(prob, DFBDF(); save_idxs = xidx)
        subsys = SciMLBase.SavedSubsystem(sys, prob.p, [xidx])
        # FIXME: hack for save_idxs
        SciMLBase.@reset dae_sol.saved_subsystem = subsys

        @brownian a b
        @mtkbuild sys = System([D(x) ~ x + p * y + x * a, D(y) ~ 2p + x^2 + y * b], t)
        xidx = variable_index(sys, x)
        prob = SDEProblem(sys, [x => 1.0, y => 2.0], (0.0, 1.0), [p => 2.0])
        sde_sol = solve(prob, SOSRI(); save_idxs = xidx)
        subsys = SciMLBase.SavedSubsystem(sys, prob.p, [xidx])
        # FIXME: hack for save_idxs
        SciMLBase.@reset sde_sol.saved_subsystem = subsys

        for sol in [ode_sol, dae_sol, sde_sol]
            prob = sol.prob
            subsys = sol.saved_subsystem
            xvals = sol[x]
            @test sol[x] == xvals
            @test is_parameter(sol, p)
            @test parameter_index(sol, p) == parameter_index(sys, p)
            @test isequal(only(parameter_symbols(sol)), p)
            @test is_independent_variable(sol, t)

            tmp = copy(prob.u0)
            tmp[xidx] = xvals[2]
            @test state_values(sol, 2) == tmp
            @test state_values(sol) == [state_values(sol, i) for i in 1:length(sol)]
        end
    end

    @testset "ODE with callbacks" begin
        @variables x(t) y(t)
        @parameters p q(t) r(t) s(t) u(t)
        evs = [0.1 => [q ~ q + 1, s ~ s - 1], 0.3 => [r ~ 2r, u ~ u / 2]]
        @mtkbuild sys = ODESystem([D(x) ~ x + p * y, D(y) ~ 2p + x], t, [x, y],
            [p, q, r, s, u], discrete_events = evs)
        @test length(unknowns(sys)) == 2
        @test length(parameters(sys)) == 5
        @test is_timeseries_parameter(sys, q)
        @test is_timeseries_parameter(sys, r)
        xidx = variable_index(sys, x)
        qidx = parameter_index(sys, q)
        qpidx = timeseries_parameter_index(sys, q)
        ridx = parameter_index(sys, r)
        rpidx = timeseries_parameter_index(sys, r)
        sidx = parameter_index(sys, s)
        uidx = parameter_index(sys, u)

        prob = ODEProblem(sys, [x => 1.0, y => 1.0], (0.0, 5.0),
            [p => 0.5, q => 0.0, r => 1.0, s => 10.0, u => 4096.0])

        @test SciMLBase.SavedSubsystem(sys, prob.p, [x, y, q, r, s, u]) === nothing

        sol = solve(prob; save_idxs = xidx)
        xvals = sol[x]
        subsys = SciMLBase.SavedSubsystem(sys, prob.p, [x, q, r])
        qvals = sol.ps[q]
        rvals = sol.ps[r]
        # FIXME: hack for save_idxs
        SciMLBase.@reset sol.saved_subsystem = subsys
        discq = DiffEqArray(SciMLBase.TupleOfArraysWrapper.(tuple.(Base.vect.(qvals))),
            sol.discretes[qpidx.timeseries_idx].t, (1, 1))
        discr = DiffEqArray(SciMLBase.TupleOfArraysWrapper.(tuple.(Base.vect.(rvals))),
            sol.discretes[rpidx.timeseries_idx].t, (1, 1))
        SciMLBase.@reset sol.discretes.collection[qpidx.timeseries_idx] = discq
        SciMLBase.@reset sol.discretes.collection[rpidx.timeseries_idx] = discr

        @test sol[x] == xvals

        @test all(Base.Fix1(is_parameter, sol), [p, q, r, s, u])
        @test all(Base.Fix1(is_timeseries_parameter, sol), [q, r])
        @test all(!Base.Fix1(is_timeseries_parameter, sol), [s, u])
        @test timeseries_parameter_index(sol, q) ==
              ParameterTimeseriesIndex(qpidx.timeseries_idx, (1, 1))
        @test timeseries_parameter_index(sol, r) ==
              ParameterTimeseriesIndex(rpidx.timeseries_idx, (1, 1))
        @test sol[q] == qvals
        @test sol[r] == rvals
    end

    @testset "SavedSubsystemWithFallback" begin
        @variables x(t) y(t)
        @parameters p q(t) r(t) s(t) u(t)
        evs = [0.1 => [q ~ q + 1, s ~ s - 1], 0.3 => [r ~ 2r, u ~ u / 2]]
        @mtkbuild sys = ODESystem([D(x) ~ x + p * y, D(y) ~ 2p + x^2], t, [x, y],
            [p, q, r, s, u], discrete_events = evs)
        prob = ODEProblem(sys, [x => 1.0, y => 1.0], (0.0, 5.0),
            [p => 0.5, q => 0.0, r => 1.0, s => 10.0, u => 4096.0])
        ss = SciMLBase.SavedSubsystem(sys, prob.p, [x, q, s, r])
        sswf = SciMLBase.SavedSubsystemWithFallback(ss, sys)
        xidx = variable_index(sys, x)
        qidx = parameter_index(sys, q)
        qpidx = timeseries_parameter_index(sys, q)
        ridx = parameter_index(sys, r)
        rpidx = timeseries_parameter_index(sys, r)
        sidx = parameter_index(sys, s)
        uidx = parameter_index(sys, u)
        @test qpidx.timeseries_idx in ss.identity_partitions
        @test !(rpidx.timeseries_idx in ss.identity_partitions)
        @test timeseries_parameter_index(sswf, q) == timeseries_parameter_index(sys, q)
        @test timeseries_parameter_index(sswf, s) == timeseries_parameter_index(sys, s)

        ptc = SciMLBase.create_parameter_timeseries_collection(sswf, prob.p, prob.tspan)
        origptc = SciMLBase.create_parameter_timeseries_collection(sys, prob.p, prob.tspan)

        @test ptc[qpidx.timeseries_idx] == origptc[qpidx.timeseries_idx]
        @test eltype(ptc[rpidx.timeseries_idx].u) <: SciMLBase.TupleOfArraysWrapper
        vals = SciMLBase.get_saveable_values(sswf, prob.p, qpidx.timeseries_idx)
        origvals = SciMLBase.get_saveable_values(sys, prob.p, qpidx.timeseries_idx)
        @test typeof(vals) == typeof(origvals)
        @test !(vals isa SciMLBase.TupleOfArraysWrapper)

        vals = SciMLBase.get_saveable_values(sswf, prob.p, rpidx.timeseries_idx)
        @test vals isa SciMLBase.TupleOfArraysWrapper
        @test vals.x isa Tuple{Vector{Float64}}
        @test vals.x[1][1] == prob.ps[r]

        vals[(1, 1)] = 2prob.ps[r]
        newp = with_updated_parameter_timeseries_values(sswf,
            parameter_values(ptc), rpidx.timeseries_idx => vals)
        @test newp[ridx] == 2prob.ps[r]
    end
end
