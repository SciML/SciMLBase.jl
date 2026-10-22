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
    # Mixed u0 shapes must widen the container (responsible_map / EnsembleSerial
    # and tmap / EnsembleThreads).
    make_prob(u0) = ODEProblem((u, p, t) -> u, u0, (0.0, 1.0), nothing)
    u0s = [0.5, diagm([1.0, 1.0])]
    probs = responsible_map(make_prob, u0s)
    @test length(probs) == 2
    @test probs[1].u0 isa Float64
    @test probs[2].u0 isa Matrix{Float64}
    tprobs = tmap(make_prob, u0s)
    @test length(tprobs) == 2
    @test tprobs[1].u0 isa Float64
    @test tprobs[2].u0 isa Matrix{Float64}
    sols = responsible_map(
        u0 -> SciMLBase.build_solution(
            make_prob(u0), :NoAlgorithm, [0.0, 1.0], [u0, u0];
            retcode = ReturnCode.Success
        ),
        u0s
    )
    @test length(sols) == 2
    @test sols[1].prob.u0 isa Float64
    @test sols[2].prob.u0 isa Matrix{Float64}
end

@testset "tmap homogeneous return type is concrete" begin
    # EnsembleThreads solve goes through tmap; a runtime misfit branch widens
    # the inferred return to abstract Vector and makes solve infer as Any.
    h(i) = Float64(i)
    @test isconcretetype(only(Base.return_types(tmap, Tuple{typeof(h), UnitRange{Int}})))
end

@testset "static array totallength" begin
    @test totallength(SVector(1.0, 2.0, 3.0)) == 3
    @test totallength(SMatrix{2, 2}(1.0, 2.0, 3.0, 4.0)) == 4
    @test iszero(@allocated totallength(SVector(1.0, 2.0, 3.0)))
end

@testset "convenience constructor iip static dispatch" begin
    f!(du, u, p, t) = (du .= u)
    f_oop(u, p, t) = u
    f1!(dv, v, u, p, t) = (dv .= v)
    f2!(du, v, u, p, t) = (du .= v)
    g!(du, u, p, t) = (du .= u)
    bc!(resid, u, p, t) = (resid[1] = u[1][1] - 1)
    A!(y, x, p, t) = (y .= x)
    nf(u, p) = u .- p
    u0 = [1.0]
    tspan = (0.0, 1.0)
    p = 1.0

    # Julia 1.10's `@allocations` expands inline with `:force_compile`, so a bare
    # constructor call does not warm the timed frame. Barrier helpers match 1.11+
    # `Base.allocations(f, args...)` and drop the compiling first measurement.
    odef_allocs(f) = @allocations ODEFunction(f)
    odep_allocs(f, u0, tspan, p) = @allocations ODEProblem(f, u0, tspan, p)
    nlp_allocs(f, u0, p) = @allocations NonlinearProblem(f, u0, p)
    sdep_allocs(f, g, u0, tspan) = @allocations SDEProblem(f, g, u0, tspan)
    bvp_allocs(f, bc, u0, tspan) = @allocations BVProblem(f, bc, u0, tspan)
    disc_allocs(f, u0, tspan) = @allocations DiscreteProblem(f, u0, tspan)
    splitf_allocs(f1, f2) = @allocations SplitFunction(f1, f2)
    dynf_allocs(f1, f2) = @allocations DynamicalODEFunction(f1, f2)
    linp_allocs(A, b) = @allocations LinearProblem(A, b)

    odef_allocs(f!)
    odef_allocs(f_oop)
    odep_allocs(f!, u0, tspan, p)
    odep_allocs(f_oop, u0, tspan, p)
    nlp_allocs(nf, u0, p)
    sdep_allocs(f!, g!, u0, tspan)
    bvp_allocs(f!, bc!, u0, tspan)
    disc_allocs(f!, u0, tspan)
    splitf_allocs(f!, f!)
    dynf_allocs(f1!, f2!)
    linp_allocs(A!, u0)

    @test odef_allocs(f!) < 40
    @test odef_allocs(f_oop) < 40
    @test odep_allocs(f!, u0, tspan, p) < 40
    @test odep_allocs(f_oop, u0, tspan, p) < 40
    @test nlp_allocs(nf, u0, p) < 40
    @test sdep_allocs(f!, g!, u0, tspan) < 50
    @test bvp_allocs(f!, bc!, u0, tspan) < 60
    @test disc_allocs(f!, u0, tspan) < 22
    @test splitf_allocs(f!, f!) < 45
    @test dynf_allocs(f1!, f2!) < 30
    @test linp_allocs(A!, u0) < 22

    @test ODEFunction(f!) isa ODEFunction{true}
    @test ODEFunction(f_oop) isa ODEFunction{false}
    @test ODEProblem(f!, u0, tspan, p).f isa ODEFunction{true}
    @test ODEProblem(f_oop, u0, tspan, p).f isa ODEFunction{false}
    @test NonlinearProblem(nf, u0, p).f isa NonlinearFunction{false}
    @test SDEProblem(f!, g!, u0, tspan).f isa SDEFunction{true}
    @test BVProblem(f!, bc!, u0, tspan).f isa BVPFunction{true}
    @test DiscreteProblem(f!, u0, tspan).f isa DiscreteFunction{true}
    @test SplitFunction(f!, f!) isa SplitFunction{true}
    @test DynamicalODEFunction(f1!, f2!) isa DynamicalODEFunction{true}
    @test isinplace(LinearProblem(A!, u0))
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
