"""
    specialize_parameters(f, ::Type{P})

Return a function that behaves as `f` does and reads the parameter object of an integrator
at the concrete type `P`.

A callback of a problem with [`DespecializedParameters`](@ref) reads `integrator.p` as an
`Any`, and each read costs a dynamic dispatch. [`erase_callback`](@ref) knows the
integrator's concrete parameter type and calls this function to restore it.

The fallback returns `f`. A package whose callback functions read parameters adds a method
that returns a function taking the parameters as a `P` when the integrator holds a `P`, and
calling `f` otherwise.

The method must keep the branch that calls `f`. `P` is the parameter type when the callback
is erased, and the integrator can hold parameters of another type afterwards: `integrator.p`
is assignable, and an erased callback is reusable with another parameter object.
"""
specialize_parameters(f, ::Type{P}) where {P} = f

"""
    CallbackTrampoline{R, I}

A function of an integrator of type `I` that returns an `R`, which hides the wrapped function's
type.
"""
const CallbackTrampoline{R, I} = FunctionWrappers.FunctionWrapper{R, Tuple{I}}

"""
$(TYPEDEF)

The condition of an [`ErasedDiscreteCallback`](@ref). It evaluates the condition of the
original callback at the state and time of the integrator.
"""
struct ErasedCondition{I}
    call::CallbackTrampoline{Bool, I}
end

(condition::ErasedCondition{I})(u, t, integrator::I) where {I} = condition.call(integrator)

"""
$(TYPEDEF)

The affect of an erased callback: `affect!` of an [`ErasedDiscreteCallback`](@ref), and
`affect!` and `affect_neg!` of an erased `ContinuousCallback`. It calls the original affect
on an integrator of type `I`.
"""
struct ErasedAffect{I}
    call::CallbackTrampoline{Nothing, I}
end

(affect!::ErasedAffect{I})(integrator::I) where {I} = affect!.call(integrator)

"""
$(TYPEDEF)

The affect of an erased [`VectorContinuousCallback`](@ref). It takes an integrator of type
`I` and the events as an `E`.
"""
struct ErasedVectorAffect{I, E}
    call::FunctionWrappers.FunctionWrapper{Nothing, Tuple{I, E}}
end

(affect!::ErasedVectorAffect{I, E})(integrator::I, events::E) where {I, E} =
    affect!.call(integrator, events)

"""
    erase_affect(affect!, ::Type{I}, ::Type{P})
    erase_affect(affect!, ::Type{I}, ::Type{P}, ::Type{E})

The [`ErasedAffect`](@ref) that calls `affect!` on an integrator of type `I`. With `E`,
the [`ErasedVectorAffect`](@ref) that calls `affect!` on the integrator and events of type
`E`.

`affect!` is first passed through [`specialize_parameters`](@ref) with `P`. An absent
affect, `nothing`, stays `nothing`.
"""
function erase_affect(affect!, ::Type{I}, ::Type{P}) where {I, P}
    specialized! = specialize_parameters(affect!, P)
    return ErasedAffect{I}(
        function (integrator)
            specialized!(integrator)
            return nothing
        end
    )
end
function erase_affect(affect!, ::Type{I}, ::Type{P}, ::Type{E}) where {I, P, E}
    specialized! = specialize_parameters(affect!, P)
    return ErasedVectorAffect{I, E}(
        function (integrator, events)
            specialized!(integrator, events)
            return nothing
        end
    )
end

erase_affect(::Nothing, ::Type, ::Type...) = nothing
erase_affect(affect!::ErasedAffect{I}, ::Type{I}, ::Type{P}) where {I, P} = affect!
erase_affect(::ErasedAffect, ::Type{I}, ::Type{P}) where {I, P} =
    throw_erased_for_another_integrator()
erase_affect(
    affect!::ErasedVectorAffect{I, E}, ::Type{I}, ::Type{P}, ::Type{E}
) where {I, P, E} = affect!
erase_affect(::ErasedVectorAffect, ::Type{I}, ::Type{P}, ::Type{E}) where {I, P, E} =
    throw_erased_for_another_integrator()

# The trampolines of an erased callback accept one integrator type, and the original
# callback is out of reach.
function throw_erased_for_another_integrator()
    throw(
        ArgumentError(
            "The callback is erased for an integrator of another type, as a callback " *
                "read from `integrator.opts.callback` is. Pass the original callback."
        )
    )
end

"""
$(TYPEDEF)

An `initialize` or `finalize` hook that trampolines to the original hook, erasing the
identity underneath. `default` records whether the original is the default hook.
"""
struct ErasedHook{I}
    call::CallbackTrampoline{Nothing, I}
    default::Bool
end

(hook::ErasedHook{I})(callback, u, t, integrator::I) where {I} = hook.call(integrator)

# A solver that stores callbacks as given calls an erased callback without passing it
# through `erase_callback`, so each erased function also rejects other integrators itself.
(::ErasedCondition)(u, t, integrator) = throw_erased_for_another_integrator()
(::ErasedAffect)(integrator) = throw_erased_for_another_integrator()
(::ErasedVectorAffect)(integrator, events) = throw_erased_for_another_integrator()
(::ErasedHook)(callback, u, t, integrator) = throw_erased_for_another_integrator()

"""
    erase_hook(hook, callback, ::Type{I})

The [`ErasedHook`](@ref) that calls `hook` on `callback` and an integrator of type `I`.
"""
function erase_hook(hook, callback, ::Type{I}) where {I}
    return ErasedHook{I}(
        function (integrator)
            hook(callback, integrator.u, integrator.t, integrator)
            return nothing
        end,
        false
    )
end

# A default hook reads nothing of its callback, so every callback shares one trampoline per
# integrator type.
erase_hook(::typeof(INITIALIZE_DEFAULT), callback, ::Type{I}) where {I} =
    ErasedHook{I}(erased_initialize_default, true)
erase_hook(::typeof(FINALIZE_DEFAULT), callback, ::Type{I}) where {I} =
    ErasedHook{I}(erased_finalize_default, true)

function erased_initialize_default(integrator)
    derivative_discontinuity!(integrator, false)
    return nothing
end
erased_finalize_default(integrator) = nothing

erase_hook(hook::ErasedHook{I}, callback, ::Type{I}) where {I} = hook
erase_hook(::ErasedHook, callback, ::Type{I}) where {I} =
    throw_erased_for_another_integrator()

is_default_finalize(hook::ErasedHook) = hook.default

# A `FunctionWrapper` holds a raw pointer to the function it wraps, and the generic
# `deepcopy` leaves that pointer on the original. These methods belong upstream, as a
# `deepcopy_internal` for `FunctionWrapper` in FunctionWrappers.jl.
copy_trampoline(call::W, copies::IdDict) where {W <: FunctionWrappers.FunctionWrapper} =
    W(Base.deepcopy_internal(call.obj[], copies))

Base.deepcopy_internal(condition::ErasedCondition{I}, copies::IdDict) where {I} =
    ErasedCondition{I}(copy_trampoline(condition.call, copies))
Base.deepcopy_internal(affect!::ErasedAffect{I}, copies::IdDict) where {I} =
    ErasedAffect{I}(copy_trampoline(affect!.call, copies))
Base.deepcopy_internal(affect!::ErasedVectorAffect{I, E}, copies::IdDict) where {I, E} =
    ErasedVectorAffect{I, E}(copy_trampoline(affect!.call, copies))
Base.deepcopy_internal(hook::ErasedHook{I}, copies::IdDict) where {I} =
    ErasedHook{I}(copy_trampoline(hook.call, copies), hook.default)

"""
    ErasedDiscreteCallback{I}

The type of every `DiscreteCallback` that [`erase_callback`](@ref) returns for an integrator
of type `I`. Its condition, affect and hooks are trampolines, so code compiled for `I` runs
every discrete callback without recompiling.
"""
const ErasedDiscreteCallback{I} = DiscreteCallback{
    ErasedCondition{I}, ErasedAffect{I}, ErasedHook{I}, ErasedHook{I}, Any, Vector{Int},
}

"""
    erase_callback(callback::DiscreteCallback, ::Type{I}, ::Type{P})

An [`ErasedDiscreteCallback`](@ref) that behaves as `callback` does on an integrator of
type `I` whose parameter object is a `P`.

The condition, the affect and the hooks are [`CallbackTrampoline`](@ref)s into `callback`.
The condition and the affect are first passed through [`specialize_parameters`](@ref)
with `P`.
"""
function erase_callback(callback::DiscreteCallback, ::Type{I}, ::Type{P}) where {I, P}
    condition = specialize_parameters(callback.condition, P)
    return ErasedDiscreteCallback{I}(
        ErasedCondition{I}(
            function (integrator)
                return condition(integrator.u, integrator.t, integrator)::Bool
            end
        ),
        erase_affect(callback.affect!, I, P),
        erase_hook(callback.initialize, callback, I),
        erase_hook(callback.finalize, callback, I),
        callback.save_positions, callback.initializealg,
        Int[callback.saved_clock_partitions...],
        callback.initialize_save_discretes
    )
end

erase_callback(callback::ErasedDiscreteCallback{I}, ::Type{I}, ::Type{P}) where {I, P} =
    callback

erase_callback(::ErasedDiscreteCallback, ::Type{I}, ::Type{P}) where {I, P} =
    throw_erased_for_another_integrator()

"""
    erase_callback(callback::ContinuousCallback, ::Type{I}, ::Type{P})
    erase_callback(callback::VectorContinuousCallback, ::Type{I}, ::Type{P}, ::Type{E})

`callback` with its affects and hooks erased for an integrator of type `I` whose parameter
object is a `P`. The affect of a `VectorContinuousCallback` takes its events as an `E`.

The condition is passed through [`specialize_parameters`](@ref) with `P` and keeps its
type: a solver calls a continuous condition at states and times of its own, and wraps the
condition for the signatures it uses.
"""
function erase_callback(callback::ContinuousCallback, ::Type{I}, ::Type{P}) where {I, P}
    return ConstructionBase.setproperties(
        callback, (;
            condition = specialize_parameters(callback.condition, P),
            affect! = erase_affect(callback.affect!, I, P),
            affect_neg! = erase_affect(callback.affect_neg!, I, P),
            initialize = erase_hook(callback.initialize, callback, I),
            finalize = erase_hook(callback.finalize, callback, I),
        )
    )
end
function erase_callback(
        callback::VectorContinuousCallback, ::Type{I}, ::Type{P}, ::Type{E}
    ) where {I, P, E}
    return ConstructionBase.setproperties(
        callback, (;
            condition = specialize_parameters(callback.condition, P),
            affect! = erase_affect(callback.affect!, I, P, E),
            initialize = erase_hook(callback.initialize, callback, I),
            finalize = erase_hook(callback.finalize, callback, I),
        )
    )
end
