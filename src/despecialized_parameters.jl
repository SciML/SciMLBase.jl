"""
$(TYPEDEF)

Store a parameter object behind the stable field type `Any`. This prevents callers such
as solver internals from specializing on the concrete parameter-container layout. Use
[`invoke_with_despecialized_parameters`](@ref) to cross a dynamic function barrier before
calling code that needs the original concrete parameter type.

This strategy accepts arbitrary parameter objects and forwards the common `Base`,
SciMLStructures, SymbolicIndexingInterface, ArrayInterface, and Adapt interfaces to the
wrapped value. The dynamic dispatch isolates inference loss at the function barrier, but
adds runtime overhead. Use [`unwrap_parameters`](@ref) in transformations such as AD that
need to reconstruct a problem with concrete parameters. Use [`FullSpecialize`](@ref) when
runtime performance takes priority over compilation reuse.

`DespecializedParameters` is distinct from [`AutoRespecialize`](@ref): the latter lets a
supporting solver recover a concrete parameter type without dynamic dispatch, but imposes
solver-specific restrictions on symbolic indexing and parameter differentiation.

The wrapped object is available through the `params` field. Other properties and
collection operations are forwarded to it.

# Fields

$(TYPEDFIELDS)
"""
struct DespecializedParameters
    params::Any
end

DespecializedParameters(params::DespecializedParameters) = params

"""
    unwrap_parameters(params)

Return the parameter object stored in [`DespecializedParameters`](@ref). The fallback is
the identity, so downstream transformations can call this function without first testing
whether parameters were despecialized.
"""
unwrap_parameters(params) = params
unwrap_parameters(params::DespecializedParameters) = getfield(params, :params)

function Base.getproperty(params::DespecializedParameters, name::Symbol)
    name === :params && return getfield(params, :params)
    return getproperty(getfield(params, :params), name)
end

function Base.propertynames(params::DespecializedParameters, private::Bool = false)
    return (:params, propertynames(getfield(params, :params), private)...)
end

Base.copy(params::DespecializedParameters) = DespecializedParameters(copy(params.params))
Base.:(==)(a::DespecializedParameters, b::DespecializedParameters) = a.params == b.params
Base.isequal(a::DespecializedParameters, b::DespecializedParameters) =
    isequal(a.params, b.params)
Base.hash(params::DespecializedParameters, h::UInt) = hash(params.params, h)

Base.IndexStyle(::Type{DespecializedParameters}) = IndexLinear()
Base.eltype(::Type{DespecializedParameters}) = Any
Base.getindex(params::DespecializedParameters, idx...) = getindex(params.params, idx...)
Base.setindex!(params::DespecializedParameters, val, idx...) =
    setindex!(params.params, val, idx...)
Base.length(params::DespecializedParameters) = length(params.params)
Base.size(params::DespecializedParameters) = size(params.params)
Base.axes(params::DespecializedParameters) = axes(params.params)
Base.firstindex(params::DespecializedParameters) = firstindex(params.params)
Base.lastindex(params::DespecializedParameters) = lastindex(params.params)
Base.iterate(params::DespecializedParameters) = iterate(params.params)
Base.iterate(params::DespecializedParameters, state) = iterate(params.params, state)

ArrayInterface.ismutable(params::DespecializedParameters) =
    ArrayInterface.ismutable(params.params)

SciMLStructures.isscimlstructure(params::DespecializedParameters) =
    SciMLStructures.isscimlstructure(params.params)
function SciMLStructures.ismutablescimlstructure(params::DespecializedParameters)
    return applicable(SciMLStructures.ismutablescimlstructure, params.params) &&
        SciMLStructures.ismutablescimlstructure(params.params)
end
function SciMLStructures.hasportion(
        portion::SciMLStructures.AbstractPortion, params::DespecializedParameters
    )
    return SciMLStructures.hasportion(portion, params.params)
end
function SciMLStructures.canonicalize(
        portion::SciMLStructures.AbstractPortion, params::DespecializedParameters
    )
    values, repack, aliases = SciMLStructures.canonicalize(portion, params.params)
    repack === nothing && return values, nothing, aliases
    return values, values -> DespecializedParameters(repack(values)), aliases
end
function SciMLStructures.replace(
        portion::SciMLStructures.AbstractPortion, params::DespecializedParameters, values
    )
    return DespecializedParameters(SciMLStructures.replace(portion, params.params, values))
end
function SciMLStructures.replace!(
        portion::SciMLStructures.AbstractPortion, params::DespecializedParameters, values
    )
    return SciMLStructures.replace!(portion, params.params, values)
end

SymbolicIndexingInterface.parameter_values(params::DespecializedParameters) = params
function SymbolicIndexingInterface.parameter_values(
        params::DespecializedParameters, idx, args...
    )
    return parameter_values(params.params, idx, args...)
end
function SymbolicIndexingInterface.set_parameter!(
        params::DespecializedParameters, value, idx
    )
    return set_parameter!(params.params, value, idx)
end
function SymbolicIndexingInterface.remake_buffer(
        indp, params::DespecializedParameters, idxs, values
    )
    return DespecializedParameters(remake_buffer(indp, params.params, idxs, values))
end
function SymbolicIndexingInterface.with_updated_parameter_timeseries_values(
        indp, params::DespecializedParameters, args...
    )
    updated = with_updated_parameter_timeseries_values(indp, params.params, args...)
    return DespecializedParameters(updated)
end

adapt_structure(to, params::DespecializedParameters) =
    DespecializedParameters(adapt(to, params.params))

Base.@noinline _invoke_with_despecialized_parameters(f, args...) = f(args...)

"""
    invoke_with_despecialized_parameters(f, args, params, ::Val{parameter_index})

Call `f(args...)` after replacing `args[parameter_index]` with the object wrapped by
`params::DespecializedParameters`. The call crosses a non-inlined dynamic-dispatch
barrier: code above the barrier sees the stable outer parameter type, while `f` compiles
for the concrete wrapped parameter type.

This is a developer interface for SciML function wrappers whose parameter position is
known statically. `args[parameter_index]` must be the same wrapper passed as `params`.
"""
Base.@inline @generated function invoke_with_despecialized_parameters(
        f, args::Tuple{Vararg{Any, N}}, params::DespecializedParameters,
        ::Val{PIdx}
    ) where {N, PIdx}
    1 <= PIdx <= N || error("parameter_index must be between 1 and $N, got $PIdx")
    call_args = [i == PIdx ? :(getfield(params, :params)) : :(args[$i]) for i in 1:N]
    return :(_invoke_with_despecialized_parameters(f, $(call_args...)))
end
