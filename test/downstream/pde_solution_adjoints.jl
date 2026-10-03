using SciMLBase, OrdinaryDiffEq, SciMLSensitivity, Test
using ChainRulesCore: ChainRulesCore, NoTangent, Tangent
using ForwardDiff: ForwardDiff
using Zygote: Zygote

# A minimal time-dependent discretizer: the states are the values of one field `:u` at the
# grid points `0, 1/(n - 1), ..., 1`, and the wrapper stores the field with time first.
struct GridMetadata <: SciMLBase.AbstractDifferentiableDiscretizationMetadata{Val(true)}
    n::Int
end
# The same discretizer, opted in without the hooks, and not opted in.
struct HooklessMetadata <: SciMLBase.AbstractDifferentiableDiscretizationMetadata{Val(true)}
    n::Int
end
struct BareMetadata <: SciMLBase.AbstractDiscretizationMetadata{Val(true)}
    n::Int
end
const Grids = Union{GridMetadata, HooklessMetadata, BareMetadata}
const GridSolution{T, N, S} = SciMLBase.PDETimeSeriesSolution{T, N, S, <:Grids}

function SciMLBase.PDETimeSeriesSolution(sol::SciMLBase.AbstractODESolution, md::Grids)
    field = reduce(vcat, (permutedims(u) for u in sol.u))
    u = Dict(:u => field)
    return SciMLBase.PDETimeSeriesSolution{
        eltype(field), 2, typeof(u), typeof(md), typeof(sol), Nothing, typeof(sol.t),
        Nothing, Vector{Symbol}, Vector{Symbol}, typeof(sol.prob), typeof(sol.alg),
        Nothing, typeof(sol.stats),
    }(
        u, sol, nothing, sol.t, nothing, [:t, :x], [:u], md, sol.prob, sol.alg, nothing,
        false, 0, sol.retcode, sol.stats
    )
end

Base.getindex(A::GridSolution, sym::Symbol) = sym === :t ? A.t : A.u[sym]
Base.getindex(A::GridSolution, sym::Symbol, inds...) = A[sym][inds...]

# The cell of a coordinate and the weight of its right end.
function cell(n, ξ)
    h = 1 / (n - 1)
    i = min(floor(Int, ξ / h) + 1, n - 1)
    return i, ξ / h - (i - 1)
end

# Evaluation at a saved time, linear in the coordinate.
function (A::GridSolution)(t::Number, ξ::Number; dv = :u)
    field = A[dv]
    k = findfirst(==(t), A.t)
    i, w = cell(A.disc_data.n, ξ)
    return (1 - w) * field[k, i] + w * field[k, i + 1]
end

# The hooks: a cotangent of the field goes back to the state vectors, one per saved time,
# and an evaluation also has a cotangent for its coordinate.
function SciMLBase.pde_index_cotangent(
        config, A::SciMLBase.PDETimeSeriesSolution{T, N, S, GridMetadata}, sym, Δ
    ) where {T, N, S}
    sym === :u || return NoTangent()
    sol = A.original_sol
    return Tangent{typeof(sol)}(; u = [Δ[k, :] for k in eachindex(sol.u)])
end

function SciMLBase.pde_call_cotangent(
        config, A::SciMLBase.PDETimeSeriesSolution{T, N, S, GridMetadata}, args, dv, Δ
    ) where {T, N, S}
    t, ξ = args
    dv = something(dv, :u)  # the default of the evaluation
    field = A[dv]
    k = findfirst(==(t), A.t)
    i, w = cell(A.disc_data.n, ξ)
    Δfield = zero(field)
    Δfield[k, i] = (1 - w) * Δ
    Δfield[k, i + 1] = w * Δ
    Δξ = Δ * (field[k, i + 1] - field[k, i]) * (A.disc_data.n - 1)
    return SciMLBase.pde_index_cotangent(config, A, dv, Δfield), (NoTangent(), Δξ)
end

n = 3
f(u, p, t) = -p .* u
prob = ODEProblem(f, ones(n), (0.0, 1.0), [1.0, 2.0, 3.0])
p0 = [1.0, 2.0, 3.0]
function wrapped(ps, md)
    sol = solve(remake(prob; p = ps), Tsit5(); saveat = 0.1, abstol = 1.0e-10, reltol = 1.0e-10)
    return SciMLBase.wrap_sol(sol, md)
end
md = GridMetadata(n)

@testset "the rules apply to the discretizers that opted in" begin
    ext = Base.get_extension(SciMLBase, :SciMLBaseChainRulesCoreExt)
    config = Zygote.ZygoteRuleConfig()
    rule_module(types...) = which(ChainRulesCore.rrule, types).module
    owner(config, args...) = rule_module(typeof(config), typeof.(args)...)
    for m in (GridMetadata(n), HooklessMetadata(n))
        A = wrapped(p0, m)
        @test owner(config, getindex, A, :u) === ext
        @test owner(config, getindex, A, :u, 1, :) === ext
        @test owner(config, A, 0.5, 0.3) === ext
        for W in (SciMLBase.PDETimeSeriesSolution, SciMLBase.PDENoTimeSolution)
            @test rule_module(Type{W}, typeof(A.original_sol), typeof(m)) === ext
        end
    end
    m = BareMetadata(n)
    A = wrapped(p0, m)
    @test owner(config, getindex, A, :u) !== ext
    @test owner(config, getindex, A, :u, 1, :) !== ext
    @test owner(config, A, 0.5, 0.3) !== ext
    for W in (SciMLBase.PDETimeSeriesSolution, SciMLBase.PDENoTimeSolution)
        @test rule_module(Type{W}, typeof(A.original_sol), typeof(m)) !== ext
    end
end

@testset "the rules route a cotangent to the solver's solution and the coordinates" begin
    loss(ps) = sum(abs2, wrapped(ps, md)[:u])
    @test Zygote.gradient(loss, p0)[1] ≈ ForwardDiff.gradient(loss, p0) rtol = 1.0e-6

    slice(ps) = sum(wrapped(ps, md)[:u, 3, :])
    @test Zygote.gradient(slice, p0)[1] ≈ ForwardDiff.gradient(slice, p0) rtol = 1.0e-6

    # A repeated index counts twice.
    repeated(ps) = sum(abs2, wrapped(ps, md)[:u, [3, 3], 2])
    @test Zygote.gradient(repeated, p0)[1] ≈ ForwardDiff.gradient(repeated, p0) rtol = 1.0e-6

    point(ps) = wrapped(ps, md)(0.5, 0.3)
    @test Zygote.gradient(point, p0)[1] ≈ ForwardDiff.gradient(point, p0) rtol = 1.0e-6

    coordinate(ξ) = wrapped(p0, md)(0.5, ξ)
    @test Zygote.gradient(coordinate, 0.3)[1] ≈ ForwardDiff.derivative(coordinate, 0.3)

    both(ps, ξ) = wrapped(ps, md)(0.5, ξ)
    @test Zygote.gradient(both, p0, 0.3)[1] ≈ ForwardDiff.gradient(point, p0) rtol = 1.0e-6
    @test Zygote.gradient(both, p0, 0.3)[2] ≈ ForwardDiff.derivative(coordinate, 0.3)

    # Indexing and evaluation of one solution object in the same loss.
    mixed(ps) = (sol = wrapped(ps, md); sum(abs2, sol[:u]) + sol(0.5, 0.3))
    @test Zygote.gradient(mixed, p0)[1] ≈ ForwardDiff.gradient(mixed, p0) rtol = 1.0e-6

    # A key that does not depend on the solve has no cotangent.
    grid(ps) = sum(wrapped(ps, md)[:t])
    @test Zygote.gradient(grid, p0)[1] === nothing
end

@testset "without the hooks the rules error instead of guessing" begin
    hookless = HooklessMetadata(n)
    @test_throws "not implemented for solution metadata type" Zygote.gradient(
        ps -> sum(abs2, wrapped(ps, hookless)[:u]), p0
    )
    @test_throws "not implemented for solution metadata type" Zygote.gradient(
        ps -> wrapped(ps, hookless)(0.5, 0.3), p0
    )
    # A cotangent reaching the wrapper through a field with no rule is an error as well.
    @test_throws "has no rule" Zygote.gradient(ps -> sum(abs2, wrapped(ps, md).u[:u]), p0)
end
