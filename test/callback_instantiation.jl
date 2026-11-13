using SciMLBase
using SciMLBase: instantiate_callback
using SciMLBase.ConstructionBase: setproperties
using Test

# Holds at every `n`-th call; the counter is callback state
struct EveryNth
    n::Int
    count::Base.RefValue{Int}
end
EveryNth(n) = EveryNth(n, Ref(0))
(c::EveryNth)(u, t, integrator) = (c.count[] += 1) % c.n == 0

function SciMLBase.instantiate_callback(cb::DiscreteCallback{<:EveryNth})
    return setproperties(cb; condition = EveryNth(cb.condition.n))
end

# A continuous condition with state
struct Shifted
    shift::Base.RefValue{Float64}
end
Shifted() = Shifted(Ref(0.0))
(c::Shifted)(u, t, integrator) = u - c.shift[]

function SciMLBase.instantiate_callback(cb::ContinuousCallback{<:Shifted})
    return setproperties(cb; condition = Shifted())
end

no_affect!(integrator) = nothing
init_callback(cb, u, t, integrator) = nothing

@testset "Callbacks without state are returned unchanged" begin
    discrete = DiscreteCallback((u, t, integrator) -> true, no_affect!)
    continuous = ContinuousCallback((u, t, integrator) -> u, no_affect!)
    vector = VectorContinuousCallback(
        (out, u, t, integrator) -> nothing,
        (integrator, idx) -> nothing, 2
    )
    for cb in (discrete, continuous, vector, nothing)
        @test instantiate_callback(cb) === cb
    end
    callbacks = CallbackSet(continuous, discrete, vector)
    @test instantiate_callback(callbacks) === callbacks
end

@testset "Stateful callbacks get fresh state" begin
    cb = DiscreteCallback(
        EveryNth(3), no_affect!; initialize = init_callback,
        save_positions = (true, false), initialize_save_discretes = false
    )
    cb.condition.count[] = 5

    a, b = instantiate_callback(cb), instantiate_callback(cb)
    @test a.condition.count !== cb.condition.count
    @test a.condition.count !== b.condition.count
    @test a.condition.count[] == 0
    @test a.condition.n == 3

    # Instances count independently
    @test [a.condition(nothing, 0.0, nothing) for _ in 1:3] == [false, false, true]
    @test b.condition.count[] == 0
    @test cb.condition.count[] == 5

    # The other fields are kept
    @test a.affect! === cb.affect!
    @test a.initialize === cb.initialize
    @test a.finalize === cb.finalize
    @test a.save_positions == cb.save_positions
    @test a.initializealg === cb.initializealg
    @test a.saved_clock_partitions === cb.saved_clock_partitions
    @test a.initialize_save_discretes == cb.initialize_save_discretes

    ccb = ContinuousCallback(Shifted(), no_affect!; abstol = 1.0e-12)
    ccb.condition.shift[] = 1.0
    c = instantiate_callback(ccb)
    @test c.condition.shift !== ccb.condition.shift
    @test c.condition(2.0, 0.0, nothing) == 2.0
    @test c.abstol == ccb.abstol
end

@testset "CallbackSet instantiates each callback" begin
    stateful = DiscreteCallback(EveryNth(2), no_affect!)
    stateless = DiscreteCallback((u, t, integrator) -> true, no_affect!)
    continuous = ContinuousCallback(Shifted(), no_affect!)

    callbacks = CallbackSet(continuous, stateful, stateless)
    instantiated = instantiate_callback(callbacks)
    @test instantiated isa CallbackSet
    @test only(instantiated.continuous_callbacks).condition.shift !==
        continuous.condition.shift
    @test instantiated.discrete_callbacks[1].condition.count !== stateful.condition.count
    @test instantiated.discrete_callbacks[2] === stateless

    # Vector storage is kept, and the original set is not modified
    erased = CallbackSet(Any[continuous], Any[stateful, stateless])
    instantiated = instantiate_callback(erased)
    @test instantiated.continuous_callbacks isa Vector{Any}
    @test instantiated.discrete_callbacks isa Vector{Any}
    @test instantiated.discrete_callbacks[1].condition.count !== stateful.condition.count
    @test instantiated.discrete_callbacks[2] === stateless
    @test erased.discrete_callbacks[1] === stateful
end

# Instantiation changes the condition's type
struct Tagged{T}
    tag::T
end
(c::Tagged)(u, t, integrator) = true

function SciMLBase.instantiate_callback(cb::DiscreteCallback{<:Tagged{Int}})
    return setproperties(cb; condition = Tagged(float(cb.condition.tag)))
end

@testset "Concretely typed vector storage can change type" begin
    cb = DiscreteCallback(Tagged(1), no_affect!)
    callbacks = CallbackSet((), [cb])
    instantiated = instantiate_callback(callbacks)
    @test only(instantiated.discrete_callbacks).condition === Tagged(1.0)
    @test callbacks.discrete_callbacks == [cb]
end
