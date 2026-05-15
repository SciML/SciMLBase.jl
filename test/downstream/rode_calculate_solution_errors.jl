using StochasticDiffEq, DiffEqNoiseProcess, SciMLBase
using Random
using Test

# Scalar out-of-place SDE with analytic solution, matching SDEProblemLibrary's
# `prob_sde_additive` shape.
f_additive(u, p, t) = @. p[2] / sqrt(1 + t) - u / (2 * (1 + t))
σ_additive(u, p, t) = @. p[1] * p[2] / sqrt(1 + t)
additive_analytic(u0, p, t, W) = @. u0 / sqrt(1 + t) + p[2] * (t + p[1] * W) / sqrt(1 + t)
ff_additive = SDEFunction(f_additive, σ_additive, analytic = additive_analytic)
p = (0.1, 0.05)

@testset "Scalar RODESolution calculate_solution_errors! with scalar NoiseGrid" begin
    # Exercises the `sol.W.u[i]` branch of calculate_solution_errors! on a
    # scalar-valued NoiseGrid (the formerly-broken `ntuple(Val(-1))` path).
    dt = 1.0e-2
    n = round(Int, 1 / dt)
    Random.seed!(12345)
    Wvals = [0.0; cumsum(sqrt(dt) * randn(n + 1))]
    Zvals = [0.0; cumsum(sqrt(dt) * randn(n + 1))]
    ts = collect(0:dt:(1 + dt))

    Wgrid = NoiseGrid(ts, Wvals, Zvals)
    @test ndims(Wgrid) == 1

    prob = SDEProblem(ff_additive, σ_additive, 1.0, (0.0, 1.0), p; noise = Wgrid)

    # The bug used to fire inside `solve!`'s `calculate_solution_errors!`
    # postamble when an analytic solution is present.
    sol = solve(prob, SRA(), dt = dt, adaptive = false)
    @test SciMLBase.successful_retcode(sol)
    @test haskey(sol.errors, :final)
    @test haskey(sol.errors, :l∞)
    @test haskey(sol.errors, :l2)
    @test isfinite(sol.errors[:final])
    @test isfinite(sol.errors[:l∞])
    @test isfinite(sol.errors[:l2])

    # Explicitly exercise calculate_solution_errors! on the already-built
    # solution to lock down the regression.
    empty!(sol.errors)
    SciMLBase.calculate_solution_errors!(sol; timeseries_errors = true, dense_errors = true)
    @test isfinite(sol.errors[:final])
    @test isfinite(sol.errors[:l∞])
    @test isfinite(sol.errors[:l2])
    @test isfinite(sol.errors[:L∞])
    @test isfinite(sol.errors[:L2])
end

@testset "Scalar RODESolution calculate_solution_errors! with default NoiseProcess" begin
    # The default WienerProcess-based path lands in the `AbstractDiffEqArray{T,N,nothing}`
    # branch of calculate_solution_errors!. Ensure the fix doesn't regress it.
    prob = SDEProblem(ff_additive, σ_additive, 1.0, (0.0, 1.0), p)
    sol = solve(prob, SRA(), dt = 0.01, adaptive = false)
    @test SciMLBase.successful_retcode(sol)
    @test isfinite(sol.errors[:final])
    @test isfinite(sol.errors[:l∞])
    @test isfinite(sol.errors[:l2])
end
