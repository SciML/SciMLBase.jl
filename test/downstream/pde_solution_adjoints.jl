using SciMLBase, OrdinaryDiffEq, NonlinearSolve, SciMLSensitivity, Test
using ChainRulesCore: ChainRulesCore
using DifferentiationInterface
using ADTypes
using ForwardDiff: ForwardDiff
using Enzyme: Enzyme
using Mooncake: Mooncake
using Zygote: Zygote

const ZYGOTE = AutoZygote()
const MOONCAKE = AutoMooncake()
const ENZYME = AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse), function_annotation = Enzyme.Const
)
const BACKENDS = [ZYGOTE, MOONCAKE, ENZYME]

backend_name(backend::ADTypes.AbstractADType) = string(typeof(backend).name.name)

# A minimal time-dependent discretizer: the states are the values of a field `:u` at the grid
# points `0, 1/(n - 1), ..., 1`, and a field `:v` is observed from them. The wrapper stores
# the fields with time first.
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

observe_v(u, p, t) = p[1] .* u .^ 2 .+ t

function SciMLBase.PDETimeSeriesSolution(sol::SciMLBase.AbstractODESolution, md::Grids)
    p = sol.prob.p
    u = reduce(vcat, (permutedims(uk) for uk in sol.u))
    v = reduce(vcat, (permutedims(observe_v(uk, p, t)) for (uk, t) in zip(sol.u, sol.t)))
    fields = Dict(:u => u, :v => v)
    return SciMLBase.PDETimeSeriesSolution{
        eltype(u), 2, typeof(fields), typeof(md), typeof(sol), Nothing, typeof(sol.t),
        Nothing, Vector{Symbol}, Vector{Symbol}, typeof(sol.prob), typeof(sol.alg),
        Nothing, typeof(sol.stats),
    }(
        fields, sol, nothing, sol.t, nothing, [:t, :x], [:u, :v], md, sol.prob, sol.alg,
        nothing, false, 0, sol.retcode, sol.stats
    )
end

# The saved times as a matrix too, so that `sol[sym]` infers: on Julia 1.10 Enzyme fails on a
# union of array types.
Base.getindex(A::GridSolution, sym::Symbol) = sym === :t ? reshape(A.t, :, 1) : A.u[sym]
Base.getindex(A::GridSolution, sym::Symbol, inds...) = A[sym][inds...]

# The cell of a coordinate and the weight of its right end.
function cell(n, ξ)
    h = 1 / (n - 1)
    i = min(floor(Int, ξ / h) + 1, n - 1)
    return i, ξ / h - (i - 1)
end

# Evaluation at a saved time, linear in the coordinate, at one point or several.
function (A::GridSolution)(t::Number, ξ::Number; dv = :u)
    field = A[dv]
    k = findfirst(==(t), A.t)
    i, w = cell(A.disc_data.n, ξ)
    return (1 - w) * field[k, i] + w * field[k, i + 1]
end
(A::GridSolution)(t::Number, ξs::AbstractVector; dv = :u) = [A(t, ξ; dv) for ξ in ξs]

# Coordinates scaled by a number: an argument with a number and an array in it.
struct Scaled{T} <: AbstractVector{T}
    s::T
    v::Vector{T}
end
Base.size(x::Scaled) = size(x.v)
Base.getindex(x::Scaled, i::Int) = x.s * x.v[i]

# The hooks: a cotangent of `:u` goes back to the state vectors, one per saved time, one of
# `:v` through the observed function, and an evaluation also has cotangents for its
# coordinates.
function SciMLBase.pde_index_cotangent(
        A::SciMLBase.PDETimeSeriesSolution{T, N, S, GridMetadata}, sym, Δ
    ) where {T, N, S}
    times = eachindex(A.original_sol.u)
    sym === :u && return (; u = [Δ[k, :] for k in times], observed = ())
    sym === :v && return (; u = nothing, observed = [(observe_v, k, Δ[k, :]) for k in times])
    return nothing
end

function SciMLBase.pde_call_cotangent(
        A::SciMLBase.PDETimeSeriesSolution{T, N, S, GridMetadata}, args, dv, Δ
    ) where {T, N, S}
    t, ξ = args
    dv = something(dv, :u)  # the default of the evaluation
    field = A[dv]
    k = findfirst(==(t), A.t)
    Δfield = zero(field)
    Δξ = map(ξ, Δ) do x, δ
        i, w = cell(A.disc_data.n, x)
        Δfield[k, i] += (1 - w) * δ
        Δfield[k, i + 1] += w * δ
        δ * (field[k, i + 1] - field[k, i]) * (A.disc_data.n - 1)
    end
    return SciMLBase.pde_index_cotangent(A, dv, Δfield), (nothing, Δξ)
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

@testset "the ChainRulesCore rules apply to the discretizers that opted in" begin
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

# Losses of the parameters, or of the parameters and a coordinate.
losses = [
    "sol[:u]" => ps -> sum(abs2, wrapped(ps, md)[:u]),
    "sol[:u, 3, :]" => ps -> sum(wrapped(ps, md)[:u, 3, :]),
    # A repeated index counts twice.
    "sol[:u, [3, 3], 2]" => ps -> sum(abs2, wrapped(ps, md)[:u, [3, 3], 2]),
    "sol[:u, 3, 2]" => ps -> wrapped(ps, md)[:u, 3, 2],
    "sol[:u, 3]" => ps -> wrapped(ps, md)[:u, 3],
    "sol(t, x)" => ps -> wrapped(ps, md)(0.5, 0.3),
    "sol(t, x; dv)" => ps -> wrapped(ps, md)(0.5, 0.3; dv = :u),
    "sol(t, x) and its coordinate" => x -> wrapped(x[1:3], md)(0.5, x[4]),
    # Coordinates in a view of the input, and in a range, which is a constant.
    "sol(t, xs) at a view, and its coordinate" =>
        x -> sum(wrapped(x[1:3], md)(0.5, view(x, 4:4))),
    "sol(t, xs) at a range" => ps -> sum(wrapped(ps, md)(0.5, 0:0.25:1)),
    # Indexing and evaluation of one solution object in the same loss.
    "sol[:u] and sol(t, x)" => ps -> (sol = wrapped(ps, md); sum(abs2, sol[:u]) + sol(0.5, 0.3)),
]
# The observed field, whose observed function also depends on the parameters.
observed_losses = [
    "sol[:v]" => ps -> sum(abs2, wrapped(ps, md)[:v]),
    "sol(t, x; dv = :v)" => ps -> wrapped(ps, md)(0.5, 0.3; dv = :v),
]
point(name) = endswith(name, "coordinate") ? [p0; 0.3] : p0

@testset "$(backend_name(backend))" for backend in BACKENDS
    @testset "$name" for (name, loss) in losses
        x = point(name)
        @test DifferentiationInterface.gradient(loss, backend, x) ≈
            ForwardDiff.gradient(loss, x) rtol = 1.0e-6
    end
    @testset "$name" for (name, loss) in observed_losses
        if backend === ENZYME
            # The parameter part of an observed value has no way back to the solve.
            @test_throws "observed" DifferentiationInterface.gradient(loss, backend, p0)
        else
            @test DifferentiationInterface.gradient(loss, backend, p0) ≈
                ForwardDiff.gradient(loss, p0) rtol = 1.0e-6
        end
    end
    coordinate(ξ) = wrapped(p0, md)(0.5, ξ[1])
    @test DifferentiationInterface.gradient(coordinate, backend, [0.3]) ≈
        ForwardDiff.gradient(coordinate, [0.3])
    # A key that does not depend on the solve has no cotangent.
    g = DifferentiationInterface.gradient(ps -> sum(wrapped(ps, md)[:t]), backend, p0)
    @test g === nothing || iszero(g)
end

# A stationary discretizer: the solver's solution is a `NonlinearSolution`, its states are the
# field `:u`, and `:v` is observed from them.
struct LineMetadata <: SciMLBase.AbstractDifferentiableDiscretizationMetadata{Val(false)}
    n::Int
end
const LineSolution{T, N, S} = SciMLBase.PDENoTimeSolution{T, N, S, LineMetadata}

observe_v(u, p) = p[1] .* u .^ 2

function SciMLBase.PDENoTimeSolution(sol::SciMLBase.AbstractNonlinearSolution, md::LineMetadata)
    fields = Dict(:u => copy(sol.u), :v => observe_v(sol.u, sol.prob.p))
    return SciMLBase.PDENoTimeSolution{
        eltype(sol.u), 1, typeof(fields), typeof(md), typeof(sol), Nothing, Vector{Symbol},
        Vector{Symbol}, typeof(sol.prob), typeof(sol.alg), Nothing, typeof(sol.stats),
    }(
        fields, sol, nothing, [:x], [:u, :v], md, sol.prob, sol.alg, nothing, sol.retcode,
        sol.stats
    )
end

Base.getindex(A::LineSolution, sym::Symbol) = A.u[sym]
Base.getindex(A::LineSolution, sym::Symbol, inds...) = A[sym][inds...]
(A::LineSolution)(i::Integer; dv = :u) = A[dv][i]

function SciMLBase.pde_index_cotangent(A::LineSolution, sym, Δ)
    sym === :u && return (; u = collect(Δ), observed = ())
    sym === :v && return (; u = nothing, observed = [(observe_v, collect(Δ))])
    return nothing
end

function SciMLBase.pde_call_cotangent(A::LineSolution, args, dv, Δ)
    (i,) = args
    dv = something(dv, :u)
    Δfield = zero(A[dv])
    Δfield[i] = Δ
    return SciMLBase.pde_index_cotangent(A, dv, Δfield), (nothing,)
end

# The states solve `u .^ 3 .+ u .= p`.
line = NonlinearProblem((u, p) -> u .^ 3 .+ u .- p, ones(n), [1.0, 2.0, 3.0])
function stationary(ps)
    sol = solve(remake(line; p = ps), NewtonRaphson(); abstol = 1.0e-12, reltol = 1.0e-12)
    return SciMLBase.wrap_sol(sol, LineMetadata(n))
end

@testset "a solution without time, $(backend_name(backend))" for backend in BACKENDS
    for loss in (
            ps -> sum(abs2, stationary(ps)[:u]),
            ps -> stationary(ps)[:u, 2],
            ps -> stationary(ps)(2),
        )
        @test DifferentiationInterface.gradient(loss, backend, p0) ≈
            ForwardDiff.gradient(loss, p0) rtol = 1.0e-6
    end
    observed(ps) = sum(abs2, stationary(ps)[:v])
    if backend === ENZYME
        @test_throws "observed" DifferentiationInterface.gradient(observed, backend, p0)
    else
        @test DifferentiationInterface.gradient(observed, backend, p0) ≈
            ForwardDiff.gradient(observed, p0) rtol = 1.0e-6
    end
end

# Under Mooncake the parameter part of an observed value goes back as fdata or rdata, by the
# form of the parameters and of their problem: a number or a matrix, in a `NonlinearProblem`,
# which is mutable, and in a `DAEProblem`, which is not. The solutions are built directly.
function nonlinear_solution(p)
    prob = NonlinearProblem((u, p) -> u .- 1, ones(n), p)
    sol = SciMLBase.build_solution(prob, nothing, ones(n), zeros(n))
    return SciMLBase.wrap_sol(sol, LineMetadata(n))
end
function dae_solution(p)
    prob = DAEProblem((du, u, p, t) -> du .+ u, -ones(n), ones(n), (0.0, 1.0), p)
    sol = SciMLBase.build_solution(prob, nothing, [0.0, 1.0], [ones(n), fill(0.5, n)])
    return SciMLBase.wrap_sol(sol, md)
end

@testset "Mooncake, parameters as a number or a matrix" begin
    forms = (only, x -> reshape(x, 1, 1))
    for solution in (nonlinear_solution, dae_solution), parameters in forms
        observed(x) = sum(abs2, solution(parameters(x))[:v])
        @test DifferentiationInterface.gradient(observed, MOONCAKE, [2.0]) ≈
            ForwardDiff.gradient(observed, [2.0])
    end
    # A number read from the solver's solution through the wrapper.
    direct(x) = dae_solution(only(x)).original_sol.prob.p
    @test DifferentiationInterface.gradient(direct, MOONCAKE, [2.0]) ≈ [1.0]
end

@testset "without the hooks the rules error instead of guessing" begin
    hookless = HooklessMetadata(n)
    @testset "$(backend_name(backend))" for backend in BACKENDS
        @test_throws "not implemented for solution metadata type" DifferentiationInterface.gradient(
            ps -> sum(abs2, wrapped(ps, hookless)[:u]), backend, p0
        )
        @test_throws "not implemented for solution metadata type" DifferentiationInterface.gradient(
            ps -> wrapped(ps, hookless)(0.5, 0.3), backend, p0
        )
        # A cotangent reaching the wrapper through a field with no rule is an error as well.
        @test_throws "has no rule" DifferentiationInterface.gradient(
            ps -> sum(abs2, wrapped(ps, md).u[:u]), backend, p0
        )
    end
    # An argument of mixed activity under Enzyme, a number and an array in one, has no way to
    # take the cotangent of its coordinates.
    @test_throws "cannot return the cotangent" DifferentiationInterface.gradient(
        x -> sum(wrapped(p0, md)(0.5, Scaled(x[1], x[2:3]))), ENZYME, [0.5, 0.3, 0.4]
    )
end
