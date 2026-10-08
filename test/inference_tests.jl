using SciMLBase, StaticArrays, Test, LinearAlgebra
using SciMLBase: has_kwargs, parameterless_type, remaker_of, responsible_map, tmap,
    totallength

f(du, u, p, t) = (du .= u)
const prob = ODEProblem(f, [1.0, 2.0], (0.0, 1.0), 1.0)

# These stand in for the many downstream branches that dispatch on the traits below;
# they return an `Int` only if the trait folded to a constant during inference.
fold_kwargs(p) = has_kwargs(p) ? 1 : 1.0
fold_parameterless_type(p) = parameterless_type(p) === ODEProblem ? 1 : 1.0

@testset "parameterless_type" begin
    @test parameterless_type(prob) === ODEProblem
    @test parameterless_type(typeof(prob)) === ODEProblem
    # A `UnionAll` problem type has no `.name` field, so it has to go through
    # `typename`.
    @test parameterless_type(ODEProblem) === ODEProblem
    @test parameterless_type(ODEProblem{true}) === ODEProblem
end

@testset "compile-time traits" begin
    @test only(Base.return_types(fold_kwargs, Tuple{typeof(prob)})) === Int
    @test only(Base.return_types(fold_parameterless_type, Tuple{typeof(prob)})) === Int
    @test only(Base.return_types(remaker_of, Tuple{typeof(prob)})) ===
        Type{ODEProblem{true}}
    @test isconcretetype(only(Base.return_types(remake, Tuple{typeof(prob)})))
end

@testset "ensemble map element types" begin
    @test responsible_map(x -> 2x, [1, 2, 3]) isa Vector{Int}
    @test responsible_map(+, [1, 2], [3, 4]) isa Vector{Int}
    @test tmap(x -> 2x, [1, 2, 3]) isa Vector{Int}
end

@testset "ensemble map widens when prob_func changes u0 type" begin
    # Downstream `ensemble_zero_length` maps over mixed u0 shapes; with concrete
    # ODEProblem inference the old narrowly typed Vector{T} threw on convert.
    make_prob(u0) = ODEProblem((u, p, t) -> u, u0, (0.0, 1.0), nothing)
    probs = responsible_map(make_prob, [0.5, diagm([1.0, 1.0])])
    @test length(probs) == 2
    @test probs[1].u0 isa Float64
    @test probs[2].u0 isa Matrix{Float64}
    sols = responsible_map(
        u0 -> SciMLBase.build_solution(
            make_prob(u0), :NoAlgorithm, [0.0, 1.0], [u0, u0];
            retcode = ReturnCode.Success
        ),
        [0.5, diagm([1.0, 1.0])]
    )
    @test length(sols) == 2
    @test sols[1].prob.u0 isa Float64
    @test sols[2].prob.u0 isa Matrix{Float64}
end

@testset "static array totallength" begin
    @test totallength(SVector(1.0, 2.0, 3.0)) == 3
    @test totallength(SMatrix{2, 2}(1.0, 2.0, 3.0, 4.0)) == 4
    @test iszero(@allocated totallength(SVector(1.0, 2.0, 3.0)))
end

@testset "convenience constructor iip static dispatch" begin
    f!(du, u, p, t) = (du .= u)
    f_oop(u, p, t) = u
    nf(u, p) = u .- p
    u0 = [1.0]
    tspan = (0.0, 1.0)
    p = 1.0

    ODEFunction(f!)
    ODEFunction(f_oop)
    ODEProblem(f!, u0, tspan, p)
    ODEProblem(f_oop, u0, tspan, p)
    NonlinearProblem(nf, u0, p)

    @test (@allocations ODEFunction(f!)) < 40
    @test (@allocations ODEFunction(f_oop)) < 40
    @test (@allocations ODEProblem(f!, u0, tspan, p)) < 40
    @test (@allocations ODEProblem(f_oop, u0, tspan, p)) < 40
    @test (@allocations NonlinearProblem(nf, u0, p)) < 40

    @test ODEFunction(f!) isa ODEFunction{true}
    @test ODEFunction(f_oop) isa ODEFunction{false}
    @test ODEProblem(f!, u0, tspan, p).f isa ODEFunction{true}
    @test ODEProblem(f_oop, u0, tspan, p).f isa ODEFunction{false}
    @test NonlinearProblem(nf, u0, p).f isa NonlinearFunction{false}
end

@testset "integrator iteration size" begin
    @test Base.IteratorSize(SciMLBase.DEIntegrator) === Base.SizeUnknown()
end

# Mirrors the shape of OptimizationBase's `OptimizationCache`: `u0`/`p` live in the
# `reinit_cache`, and some field is abstractly typed (`solver_args::NamedTuple`).
struct MockReInitCache{U, P}
    u0::U
    p::P
end
struct MockOptimizationCache{F, R} <: SciMLBase.AbstractOptimizationCache
    f::F
    reinit_cache::R
    solver_args::NamedTuple
end

@testset "optimization cache u0/p inference" begin
    cache = MockOptimizationCache(sin, MockReInitCache([1.0, 2.0], 2.0), (; a = 1))
    # reading the reinit cache through `getproperty` again would make the call recursive,
    # and inference would return the union of all field types instead of the field's
    @test only(Base.return_types(c -> c.p, Tuple{typeof(cache)})) === Float64
    @test only(Base.return_types(c -> c.u0, Tuple{typeof(cache)})) === Vector{Float64}
    @test only(Base.return_types(c -> c.f, Tuple{typeof(cache)})) === typeof(sin)
    @test cache.p === 2.0
    @test cache.u0 == [1.0, 2.0]
end
