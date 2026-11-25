"""
    specialize_parameters(f, ::Type{P})

Return a function that behaves as `f` does and reads the parameter object of an integrator
at the concrete type `P`. Basic (unspecialized) callbacks are called with [`DespecializedParameters`](@ref)
and access the despecialized-to-any `integrator.p`. This causes dynamic dispatch and sad.
[`erase_callback`](@ref) knows the concrete parameter type of the integrator, though, and
can use this function to recover the fully specific type.

The fallback returns `f`. A package whose callback functions read parameters adds a method
that returns a function taking the parameters as a `P` when the integrator holds a `P`, and
calling `f` otherwise.
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

The affect of an [`ErasedDiscreteCallback`](@ref).
"""
struct ErasedAffect{I}
    call::CallbackTrampoline{Nothing, I}
end

(affect!::ErasedAffect{I})(integrator::I) where {I} = affect!.call(integrator)

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

# The default hooks are no-ops so they can share the same erased hook representation.
erase_hook(::typeof(INITIALIZE_DEFAULT), callback, ::Type{I}) where {I} =
    ErasedHook{I}(erased_initialize_default, true)
erase_hook(::typeof(FINALIZE_DEFAULT), callback, ::Type{I}) where {I} =
    ErasedHook{I}(erased_finalize_default, true)

function erased_initialize_default(integrator)
    derivative_discontinuity!(integrator, false)
    return nothing
end
erased_finalize_default(integrator) = nothing

is_default_finalize(hook::ErasedHook) = hook.default

# A `FunctionWrapper` holds a raw pointer to the function it wraps, and the generic
# `deepcopy` leaves that pointer on the original. These methods belong upstream, as a
# `deepcopy_internal` for `FunctionWrapper` in FunctionWrappers.jl.
copy_trampoline(call::CallbackTrampoline{R, I}, copies::IdDict) where {R, I} =
    CallbackTrampoline{R, I}(Base.deepcopy_internal(call.obj[], copies))

Base.deepcopy_internal(condition::ErasedCondition{I}, copies::IdDict) where {I} =
    ErasedCondition{I}(copy_trampoline(condition.call, copies))
Base.deepcopy_internal(affect!::ErasedAffect{I}, copies::IdDict) where {I} =
    ErasedAffect{I}(copy_trampoline(affect!.call, copies))
Base.deepcopy_internal(hook::ErasedHook{I}, copies::IdDict) where {I} =
    ErasedHook{I}(copy_trampoline(hook.call, copies), hook.default)

"""
    ErasedDiscreteCallback{I}

The monomorphic type of every callback that [`erase_callback`](@ref) returns for an integrator of type
`I`. The callback implementations are hidden underneath the indirection so that the integrator's compiled
code doesn't become contaminated by their implementations and the precompiled code remains usable.
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
    affect! = specialize_parameters(callback.affect!, P)
    return ErasedDiscreteCallback{I}(
        ErasedCondition{I}(
            function (integrator)
                return condition(integrator.u, integrator.t, integrator)::Bool
            end
        ),
        ErasedAffect{I}(
            function (integrator)
                affect!(integrator)
                return nothing
            end
        ),
        erase_hook(callback.initialize, callback, I),
        erase_hook(callback.finalize, callback, I),
        callback.save_positions, callback.initializealg,
        Int[callback.saved_clock_partitions...],
        callback.initialize_save_discretes
    )
end

erase_callback(callback::ErasedDiscreteCallback{I}, ::Type{I}, ::Type{P}) where {I, P} =
    callback

function erase_callback(::ErasedDiscreteCallback, ::Type{I}, ::Type{P}) where {I, P}
    throw(
        ArgumentError(
            "A callback that was erased for an integrator of another type" * 
                "(such as one taken from `integrator.opts.callback`) was tried to be used for an integrator of type $I."*
                "Pass the original callback."
        )
    )
end
