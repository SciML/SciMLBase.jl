using SciMLBase
using SciMLBase: erase_callback, ErasedDiscreteCallback
using Test

mutable struct MockIntegrator <: SciMLBase.DEIntegrator{Nothing, Nothing, Vector{Float64}, Float64}
    u::Vector{Float64}
    t::Float64
    p::Any
    saved::Vector{Int}
    derivative_discontinuity::Bool
end
MockIntegrator(p) = MockIntegrator([0.0], 0.0, p, Int[], true)

struct OtherIntegrator end

function SciMLBase.save_discretes!(integrator::MockIntegrator, idx::Int; skip_duplicates = false)
    push!(integrator.saved, idx)
    return nothing
end
function SciMLBase.derivative_discontinuity!(integrator::MockIntegrator, bool::Bool)
    integrator.derivative_discontinuity = bool
    return nothing
end

struct AddParameter end
(::AddParameter)(integrator) = (integrator.u[1] += integrator.p; nothing)
struct AddParameterAt{P} end
(::AddParameterAt{P})(integrator) where {P} = (integrator.u[1] += integrator.p::P; nothing)
SciMLBase.specialize_parameters(::AddParameter, ::Type{P}) where {P} = AddParameterAt{P}()

affect_allocations(callback, integrator) = @allocated callback.affect!(integrator)

final_saves(callback) = final_saves(callback, MockIntegrator(2.0))
function final_saves(callback, integrator)
    SciMLBase.save_final_discretes!(integrator, callback)
    return integrator.saved
end

@testset "condition and affect" begin
    integrator = MockIntegrator(2.0)
    I = typeof(integrator)
    erased = erase_callback(DiscreteCallback((u, t, i) -> u[1] < 1, AddParameter()), I, Float64)
    other = erase_callback(DiscreteCallback((u, t, i) -> true, identity), I, Float64)

    @test erased isa ErasedDiscreteCallback{I}
    @test typeof(erased) === typeof(other)
    @test erased.condition(integrator.u, integrator.t, integrator)
    erased.affect!(integrator)
    @test integrator.u == [2.0]
    @test !erased.condition(integrator.u, integrator.t, integrator)
    @test affect_allocations(erased, integrator) == 0
    @test erase_callback(erased, I, Float64) === erased
    @test_throws ArgumentError erase_callback(erased, OtherIntegrator, Float64)
end

@testset "hooks" begin
    integrator = MockIntegrator(2.0)
    I = typeof(integrator)
    seen = Any[]
    record(callback, u, t, integrator) = push!(seen, callback)
    custom = DiscreteCallback((u, t, i) -> true, identity; initialize = record, finalize = record)
    erased = erase_callback(custom, I, Float64)
    erased.initialize(erased, integrator.u, integrator.t, integrator)
    erased.finalize(erased, integrator.u, integrator.t, integrator)
    @test seen == Any[custom, custom]
    @test all(callback -> callback === custom, seen)

    default = erase_callback(DiscreteCallback((u, t, i) -> true, identity), I, Float64)
    default.initialize(default, integrator.u, integrator.t, integrator)
    @test !integrator.derivative_discontinuity
    @test default.finalize(default, integrator.u, integrator.t, integrator) === nothing
end

mutable struct Counter
    count::Int
end
(counter::Counter)(integrator) = (counter.count += 1; nothing)

@testset "deepcopy" begin
    integrator = MockIntegrator(2.0)
    I = typeof(integrator)
    counter = Counter(0)
    report = (callback, u, t, integrator) -> (integrator.u[1] = callback.affect!.count)
    erased = erase_callback(
        DiscreteCallback((u, t, i) -> true, counter; finalize = report), I, Float64
    )
    copy = deepcopy(erased)
    @test copy isa ErasedDiscreteCallback{I}
    @test copy.condition(integrator.u, integrator.t, integrator)
    copy.affect!(integrator)
    copy.affect!(integrator)
    @test counter.count == 0
    # The copied hook sees the affect that the copied trampoline runs.
    copy.finalize(copy, integrator.u, integrator.t, integrator)
    @test integrator.u == [2.0]
    @test !SciMLBase.is_default_finalize(copy.finalize)
    @test SciMLBase.is_default_finalize(
        deepcopy(erase_callback(DiscreteCallback((u, t, i) -> true, identity), I, Float64)).finalize
    )
end

@testset "final discrete saving matches the original callback" begin
    I = MockIntegrator
    condition = (u, t, i) -> true
    finalize = (callback, u, t, integrator) -> nothing
    for kwargs in (
            (;), (; finalize), (; finalize, save_positions = (true, false)),
        )
        callback = DiscreteCallback(condition, identity; saved_clock_partitions = (1, 2), kwargs...)
        @test final_saves(erase_callback(callback, I, Float64)) == final_saves(callback)
    end
    default = DiscreteCallback(condition, identity; saved_clock_partitions = (1, 2))
    @test isempty(final_saves(erase_callback(default, I, Float64)))
    custom = DiscreteCallback(condition, identity; saved_clock_partitions = (1, 2), finalize)
    @test final_saves(erase_callback(custom, I, Float64)) == [1, 2]
end
