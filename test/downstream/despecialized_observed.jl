using ModelingToolkit, OrdinaryDiffEq, SymbolicIndexingInterface, SciMLBase
using ModelingToolkit: t_nounits as t, D_nounits as D
using Test

# MTK builds problems with `specialize = AutoDespecialize`, so the solution carries its
# parameters behind the `DespecializedParameters` barrier. Whole-timeseries observed
# getters must cross that barrier once (`with_concrete_parameters`) rather than once per
# saved point, so that the broadcast below the barrier infers a concrete element type.
# Before that, the getter allocated per element and assembled a widened result.
@variables x(t) y(t) z(t)
@parameters α β γ δ
eqs = [
    D(x) ~ α * x - β * x * y,
    D(y) ~ -γ * y + δ * x * y,
    z ~ x + y,
]
@mtkcompile sys = System(eqs, t)
prob = ODEProblem(
    sys, [x => 1.0, y => 1.0, α => 1.5, β => 1.0, γ => 3.0, δ => 1.0], (0.0, 10.0)
)
sol = solve(prob, Tsit5(); saveat = 0.01)
@test parameter_values(sol) isa SciMLBase.DespecializedParameters

getz = getsym(sys, z)
zvals = getz(sol)
@test zvals isa Vector{Float64}
@test zvals ≈ sol[x] .+ sol[y]
@test getz(sol, 2:5) == zvals[2:5]
@test getz(sol, 2:5) isa Vector{Float64}
mask = sol.t .< 1.0
@test getz(sol, mask) == zvals[mask]
@test getz(sol, mask) isa Vector{Float64}
@test getz(sol, 3) == zvals[3]

# Crossing the barrier per element cost tens of thousands of allocations for this many
# saved points; crossing it once costs a handful plus the result.
getz(sol)
@test (@allocated getz(sol)) < 10 * sizeof(zvals)
