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
function (::AddParameterAt{P})(integrator) where {P}
    p = integrator.p
    p isa P || return AddParameter()(integrator)
    integrator.u[1] += p
    return nothing
end
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

    # The integrator's parameters are no longer the `P` the callback was erased with.
    integrator.p = 3
    integrator.u[1] = 0.0
    erased.affect!(integrator)
    @test integrator.u == [3.0]
    @test erase_callback(erased, I, Int) === erased
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

@testset "continuous callbacks" begin
    integrator = MockIntegrator(2.0)
    I = typeof(integrator)
    up, down = Counter(0), Counter(0)
    seen = Any[]
    record = (callback, u, t, integrator) -> push!(seen, callback)
    condition = (u, t, integrator) -> u[1] - 1
    callback = ContinuousCallback(condition, up, down; initialize = record)
    erased = erase_callback(callback, I, Float64)
    other = erase_callback(ContinuousCallback(condition, identity, nothing), I, Float64)

    @test erased.condition === condition
    @test erased.affect! isa SciMLBase.ErasedAffect{I}
    @test typeof(erased.affect!) === typeof(other.affect!)
    @test other.affect_neg! === nothing
    @test typeof(erased.initialize) === typeof(erased.finalize) === typeof(other.initialize)
    erased.affect!(integrator)
    erased.affect_neg!(integrator)
    erased.affect_neg!(integrator)
    @test (up.count, down.count) == (1, 2)
    erased.initialize(erased, integrator.u, integrator.t, integrator)
    @test only(seen) === callback
    @test SciMLBase.is_default_finalize(erased.finalize)

    again = erase_callback(erased, I, Float64)
    @test again.affect! === erased.affect!
    @test again.initialize === erased.initialize
    @test_throws ArgumentError erase_callback(erased, OtherIntegrator, Float64)

    specialized = erase_callback(ContinuousCallback(condition, AddParameter()), I, Float64)
    specialized.affect!(integrator)
    @test affect_allocations(specialized, integrator) == 0
end

@testset "vector continuous callbacks" begin
    integrator = MockIntegrator(2.0)
    I = typeof(integrator)
    events = Int8[1, 0, -1]
    E = typeof(view(events, 1:2))
    fired = Int8[]
    affect! = (integrator, events) -> append!(fired, events)
    condition = (out, u, t, integrator) -> fill!(out, u[1])
    callback = VectorContinuousCallback(condition, affect!, 2)
    erased = erase_callback(callback, I, Float64, E)

    @test erased.len == 2
    @test erased.affect! isa SciMLBase.ErasedVectorAffect{I, E}
    @test erased.affect!(integrator, view(events, 1:2)) === nothing
    @test fired == Int8[1, 0]
    @test erase_callback(erased, I, Float64, E).affect! === erased.affect!
    @test_throws ArgumentError erase_callback(erased, OtherIntegrator, Float64, E)

    copy = deepcopy(erased)
    copy.affect!(integrator, view(events, 2:3))
    @test fired == Int8[1, 0]
    @test erase_callback(VectorContinuousCallback(condition, nothing, 2), I, Float64, E).affect! === nothing
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
