using StochasticDiffEq, DiffEqNoiseProcess, SciMLBase
using Random
using Test

# Regression tests for `SciMLBase.calculate_solution_errors!(::AbstractRODESolution)`.
#
# Two separate failure modes were reported on RAT v4:
#
# 1. SciMLBase#1321: `for i in 1:length(sol)` overran `sol.t` / `sol.W` because
#    RAT v4 dropped the `length(::AbstractVectorOfArray)` specialization, so
#    `length(sol)` became the total scalar count rather than the number of
#    saved time points. Fixed by switching to `for i in eachindex(sol.t)`.
#
# 2. This test file: even with the loop bounded correctly, the indexing
#    expression `sol.W[:, i]` routes through
#    `RecursiveArrayTools._getindex(::AbstractDiffEqArray, ::NotSymbolic,
#    ::Colon, ::Int)` which calls `size(::AbstractVectorOfArray)`, which on
#    RAT v4 computes `ntuple(Val(N - 1))`. For a scalar-valued `NoiseGrid`
#    built from a `Vector{<:Number}`, `N == 0` (because `ndims(W[1]) == 0`),
#    producing `ntuple(Val(-1))` and throwing
#    `ArgumentError: tuple length should be ≥ 0, got -1`.
#
# The fix replaces `sol.W[:, i]` with `sol.W.u[i]`, which is equivalent on the
# vector-valued branch and handles scalar-valued noise without consulting
# `size`.

# Scalar out-of-place SDE with analytic solution, matching SDEProblemLibrary's
# `prob_sde_additive` shape.
f_additive(u, p, t) = @. p[2] / sqrt(1 + t) - u / (2 * (1 + t))
σ_additive(u, p, t) = @. p[1] * p[2] / sqrt(1 + t)
additive_analytic(u0, p, t, W) = @. u0 / sqrt(1 + t) + p[2] * (t + p[1] * W) / sqrt(1 + t)
ff_additive = SDEFunction(f_additive, σ_additive, analytic = additive_analytic)
p = (0.1, 0.05)

@testset "Scalar RODESolution calculate_solution_errors! with NoiseGrid (N=0)" begin
    # NoiseGrid built from a Vector{Float64} has N=0 — this is the path that
    # used to hit `ntuple(Val(-1))`.
    dt = 1.0e-2
    n = round(Int, 1 / dt)
    Random.seed!(12345)
    Wvals = [0.0; cumsum(sqrt(dt) * randn(n + 1))]
    Zvals = [0.0; cumsum(sqrt(dt) * randn(n + 1))]
    ts = collect(0:dt:(1 + dt))

    Wgrid = NoiseGrid(ts, Wvals, Zvals)
    @test ndims(Wgrid) == 0  # scalar-valued NoiseGrid carries N=0

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

@testset "Scalar RODESolution calculate_solution_errors! with default NoiseProcess (N=1)" begin
    # The default WienerProcess-based path lands in the same `else` branch
    # of calculate_solution_errors! but with N=1. Ensure the fix doesn't
    # regress it.
    prob = SDEProblem(ff_additive, σ_additive, 1.0, (0.0, 1.0), p)
    sol = solve(prob, SRA(), dt = 0.01, adaptive = false)
    @test SciMLBase.successful_retcode(sol)
    @test isfinite(sol.errors[:final])
    @test isfinite(sol.errors[:l∞])
    @test isfinite(sol.errors[:l2])
end
