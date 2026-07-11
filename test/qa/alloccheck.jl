# Static allocation proof for the scalar in-place interpolation path, on top
# of the runtime `@allocated == 0` checks in test/interpolation_tests.jl:
# AllocCheck analyzes the compiled code and fails if a reachable allocation
# site exists, so a regression shows up even on inputs the runtime test does
# not exercise.
#
# Two documented exclusions, each the tightest truthful bound available:
#
# - `ignore_throw = true` is AllocCheck's documented option for genuine
#   user-facing error branches: the interpolation path's extrapolation-bound
#   errors, the sensitivitymode error, and broadcast shape checks construct
#   exceptions, which necessarily allocates. Only throw-path allocations are
#   excluded by this.
#
# - Alias-guard sites: the interpolant kernels broadcast into `out`
#   (`@. out = ...`), and Base broadcast's `preprocess` contains a
#   conditional defensive copy (`unaliascopy`) taken only when `out` aliases
#   the interpolant's own stored timeseries — required for broadcast
#   correctness and unremovable without abandoning broadcast in the kernels.
#   It never fires for a caller-owned output buffer. The test asserts that
#   every reported site IS such an alias guard (identified by `unaliascopy`
#   in its backtrace), so any unconditional allocation still fails.
using SciMLBase, AllocCheck, Test

t_ac = [0.0, 1.0, 2.0, 3.0]
u_ac = [[1.0, 2.0], [1.5, 2.4], [2.1, 2.9], [2.8, 3.5]]
du_ac = [[0.4, 0.3], [0.5, 0.4], [0.6, 0.5], [0.7, 0.6]]

# Concrete entry point matching the solution-interpolation call shape.
scalar_interp!(itp, out, tval, deriv::D) where {D} =
    itp(out, tval, nothing, deriv, nothing, :left)

is_alias_guard(site) = any(fr -> fr.func === :unaliascopy, site.backtrace)

@testset "AllocCheck: scalar in-place interpolation" begin
    # BasicInterpolation dense and non-dense share one concrete type, so a
    # single signature covers both runtime modes of the method instance.
    interps = (
        SciMLBase.HermiteInterpolation(t_ac, u_ac, du_ac),
        SciMLBase.LinearInterpolation(t_ac, u_ac),
        SciMLBase.BasicInterpolation(t_ac, u_ac, du_ac, true),
    )
    for itp in interps, deriv in (Val{0}, Val{1})
        sig = (typeof(itp), Vector{Float64}, Float64, Type{deriv})
        allocs = AllocCheck.check_allocs(scalar_interp!, sig; ignore_throw = true)
        @test all(is_alias_guard, allocs)
    end
end
