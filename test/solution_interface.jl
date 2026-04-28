using Test, SciMLBase

@testset "getindex" begin
    u = rand(1)
    @test SciMLBase.build_linear_solution(nothing, u, zeros(1), nothing)[] == u[]
end

@testset "plot ODE solution" begin
    f = ODEFunction((u, p, t) -> -u, analytic = (u0, p, t) -> u0 * exp(-t))
    ode = ODEProblem(f, 1.0, (0.0, 1.0))
    sol = SciMLBase.build_solution(ode, :NoAlgorithm, [ode.tspan[begin]], [ode.u0])
    for t in Iterators.drop(range(ode.tspan..., length = 5), 1)
        push!(sol.t, t)
        push!(sol.u, ode.u0)
    end

    int_vars = SciMLBase.interpret_vars(nothing, sol) # nothing = idxs
    plot_vecs,
        labels = SciMLBase.diffeq_to_arrays(
        sol,
        true, # plot_analytic
        true, # denseplot
        10, # plotdensity
        ode.tspan,
        int_vars,
        :identity,
        nothing
    ) # tscale
    @test plot_vecs[2][:, 2] ≈ @. exp(-plot_vecs[1][:, 2])
end

# Regression test for #1335 / OrdinaryDiffEq.jl#3573: under
# RecursiveArrayTools v4 / SciMLBase v3, `length(sol::ODESolution)` is
# `prod(size(sol)) = state_dim * n_timesteps`, not the timestep count.
# `diffeq_to_arrays` was using `end_idx = length(sol)` which then
# blew up on `sol.t[end_idx]` for any multi-state solve. The scalar
# "plot ODE solution" testset above doesn't catch this because
# state_dim == 1. This exercises the multi-state path through the
# Makie extension's `convert_arguments` entry (tspan === nothing,
# denseplot = true).
@testset "plot multi-state ODE solution (length(sol) != length(sol.t))" begin
    f = ODEFunction((du, u, p, t) -> (du .= -u))
    ode = ODEProblem(f, [1.0, 2.0, 3.0], (0.0, 1.0))
    sol = SciMLBase.build_solution(
        ode, :NoAlgorithm, [ode.tspan[begin]], [copy(ode.u0)]
    )
    for t in Iterators.drop(range(ode.tspan..., length = 8), 1)
        push!(sol.t, t)
        push!(sol.u, copy(ode.u0))
    end
    @test length(sol) != length(sol.t)  # 24 vs 8 — the precondition

    int_vars = SciMLBase.interpret_vars(nothing, sol)
    plot_vecs, labels = SciMLBase.diffeq_to_arrays(
        sol,
        false,                # plot_analytic
        true,                 # denseplot — exercises the end_idx path
        10 * length(sol.t),   # plotdensity (matches @recipe default shape)
        nothing,              # tspan === nothing — triggers `end_idx = length(sol)`
        int_vars,
        :identity,
        nothing               # plotat
    )
    @test length(plot_vecs) == 2
    @test length(labels) == 3
    @test all(size(v, 1) > 0 for v in plot_vecs)
end

@testset "interpolate empty ODE solution" begin
    f = (u, p, t) -> -u
    ode = ODEProblem(f, 1.0, (0.0, 1.0))
    sol = SciMLBase.build_solution(ode, :NoAlgorithm, [ode.tspan[begin]], [ode.u0])
    @test sol(0.0) == 1.0
    @test sol([0.0, 0.0]) == [1.0, 1.0]
    # test that indexing out of bounds doesn't segfault
    @test_throws ErrorException sol(1)
    @test_throws ErrorException sol(-0.5)
    @test_throws ErrorException sol([0, -0.5, 0])
end

@testset "interpolate with empty idxs" begin
    f = (u, p, t) -> u
    sol1 = SciMLBase.build_solution(
        ODEProblem(f, 1.0, (0.0, 1.0)), :NoAlgorithm, 0.0:0.1:1.0, exp.(0.0:0.1:1.0)
    )
    sol2 = SciMLBase.build_solution(
        ODEProblem(f, [1.0, 2.0], (0.0, 1.0)), :NoAlgorithm,
        0.0:0.1:1.0, vcat.(exp.(0.0:0.1:1.0), 2exp.(0.0:0.1:1.0))
    )
    for sol in [sol1, sol2]
        @test sol(0.15; idxs = []) == Float64[]
        @test sol(0.15; idxs = Int[]) == Float64[]
        @test sol([0.15, 0.25]; idxs = []) == [Float64[], Float64[]]
        @test sol([0.15, 0.25]; idxs = Int[]) == [Float64[], Float64[]]
    end
end

@testset "iterate uses AbstractArray fallback, not container-yielding" begin
    using LinearAlgebra
    f = (u, p, t) -> -u
    ode = ODEProblem(f, [1.0, 2.0], (0.0, 1.0))
    sol = SciMLBase.build_solution(
        ode, :NoAlgorithm, collect(0.0:0.1:1.0),
        [[exp(-t), 2exp(-t)] for t in 0.0:0.1:1.0]
    )
    first_elem, _ = iterate(sol)
    @test !(first_elem isa typeof(sol))
    # `LinearAlgebra.norm(sol)` must not throw `norm_recursive_check`.
    @test isfinite(LinearAlgebra.norm(sol))
    @test sol ≈ sol
end

@testset "solution_new_original_retcode type inference" begin
    # Regression test: `solution_new_original_retcode` must not widen the
    # returned ODESolution's type parameters through the Accessors / setproperties
    # rebuild chain even when callers annotate the `original` argument as the
    # bare `NonlinearSolution` UnionAll (see BoundaryValueDiffEqCore.__build_solution).
    # See https://github.com/SciML/BoundaryValueDiffEq.jl/issues/479.
    f = (u, p, t) -> -u
    ode = ODEProblem(f, [1.0, 2.0], (0.0, 1.0))
    odesol = SciMLBase.build_solution(
        ode, :NoAlgorithm, collect(0.0:0.1:1.0),
        [[exp(-t), 2exp(-t)] for t in 0.0:0.1:1.0],
    )

    # Build a NonlinearSolution whose 7th (O = original) type parameter is
    # pinned to `Any` --- matches NonlinearSolveBase.build_solution_less_specialize.
    u = [1.0, 2.0]
    resid = [0.0, 0.0]
    nlsol = SciMLBase.NonlinearSolution{
        Float64, 1, typeof(u), typeof(resid), Nothing, Nothing,
        Any, Nothing, Nothing, Nothing,
    }(u, resid, nothing, nothing, SciMLBase.ReturnCode.Success, [0.1, 0.2],
      nothing, nothing, nothing, nothing)

    # Direct concrete call: must be fully inferrable.
    @inferred SciMLBase.solution_new_original_retcode(
        odesol, nlsol, SciMLBase.ReturnCode.Success, resid,
    )

    # Caller with bare `::NonlinearSolution` annotation (mimicking BVDE's
    # __build_solution): the inferred return type must NOT collapse to the
    # bare `ODESolution` UnionAll.
    bvde_like(o, n::SciMLBase.NonlinearSolution) = SciMLBase.solution_new_original_retcode(
        o, n, SciMLBase.ReturnCode.Success, n.resid,
    )
    rt_bare = Base.return_types(bvde_like, (typeof(odesol), SciMLBase.NonlinearSolution))[1]
    @test rt_bare !== SciMLBase.ODESolution

    # And when the concrete nlsol type is known at the call site, inference
    # must produce a fully concrete ODESolution type.
    @inferred bvde_like(odesol, nlsol)
end
