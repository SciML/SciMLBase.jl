module SciMLBaseMooncakeExt

using SciMLBase, Mooncake
using SciMLBase: ADOriginator, ChainRulesOriginator, MooncakeOriginator
import Mooncake: rrule!!, CoDual, zero_fcodual, @is_primitive,
    @from_rrule, @zero_adjoint, @mooncake_overlay, MinimalCtx,
    NoPullback, NoTangent, NoRData, primal, tangent, prepare_pullback_cache,
    value_and_pullback!!

@zero_adjoint MinimalCtx Tuple{typeof(SciMLBase.numargs), Any}
@is_primitive MinimalCtx Tuple{
    typeof(SciMLBase.set_mooncakeoriginator_if_mooncake), SciMLBase.ChainRulesOriginator,
}

@mooncake_overlay SciMLBase.set_mooncakeoriginator_if_mooncake(x::SciMLBase.ADOriginator) = SciMLBase.MooncakeOriginator()

function rrule!!(
        f::CoDual{typeof(SciMLBase.set_mooncakeoriginator_if_mooncake)},
        X::CoDual{SciMLBase.ChainRulesOriginator}
    )
    return zero_fcodual(SciMLBase.MooncakeOriginator()), NoPullback(f, X)
end

# ============================================================================
# tmap and responsible_map rules for Ensemble AD
# These enable Mooncake to differentiate through ensemble solves by providing
# proper AD nesting for the mapped function.
# See: https://github.com/SciML/DiffEqBase.jl/issues/1256
# ============================================================================

# Mark tmap and responsible_map as primitives
@is_primitive MinimalCtx Tuple{typeof(SciMLBase.tmap), Any, Vararg}
@is_primitive MinimalCtx Tuple{typeof(SciMLBase.responsible_map), Any, Vararg}

# Helper to accumulate tangents
function _accum_tangents(a::NoTangent, b::NoTangent)
    return NoTangent()
end
function _accum_tangents(a::NoTangent, b)
    return b
end
function _accum_tangents(a, b::NoTangent)
    return a
end
function _accum_tangents(a::T, b::T) where {T <: Number}
    return a + b
end
function _accum_tangents(a::Tuple, b::Tuple)
    return map(_accum_tangents, a, b)
end
function _accum_tangents(a::NamedTuple{N}, b::NamedTuple{N}) where {N}
    return NamedTuple{N}(map(_accum_tangents, values(a), values(b)))
end
function _accum_tangents(a, b)
    # Fallback: try addition
    return a + b
end

"""
    rrule!! for SciMLBase.tmap

Implements reverse-mode AD for tmap by:
1. Forward pass: Compute primals and prepare pullback caches for each element
2. Reverse pass: Read gradients from output fdata, compute input gradients via caches

Note: For vectors, Mooncake uses fdata (tangent field of CoDual) for gradients,
not rdata. The pullback receives NoRData() and must read the gradient from
the output's fdata which was modified by the downstream operation.
"""
function rrule!!(
        ::CoDual{typeof(SciMLBase.tmap)},
        f_dual::CoDual{F},
        args_dual::CoDual...
    ) where {F}
    # Extract primals and tangents (fdata)
    f = primal(f_dual)
    args = map(primal, args_dual)
    args_tangents = map(tangent, args_dual)

    n = length(args[1])

    # Compute first element to determine output type
    if n == 0
        # Empty case - infer type from function signature if possible
        T = Core.Compiler.return_type(f, Tuple{map(eltype, args)...})
        ys = Vector{T}(undef, 0)
        return zero_fcodual(ys), NoPullback(zero_fcodual(SciMLBase.tmap), f_dual, args_dual...)
    end

    # Forward pass: compute values and prepare caches
    caches = Vector{Any}(undef, n)

    # Compute first element to get the type
    arg_1 = ntuple(j -> args[j][1], length(args))
    caches[1] = prepare_pullback_cache(f, arg_1...)
    y1 = f(arg_1...)

    # Create properly typed output vector and its tangent (fdata)
    ys = Vector{typeof(y1)}(undef, n)
    ys[1] = y1
    ys_tangent = zeros(typeof(y1), n)

    # Compute remaining elements
    for i in 2:n
        arg_i = ntuple(j -> args[j][i], length(args))
        # Prepare cache for this call
        caches[i] = prepare_pullback_cache(f, arg_i...)
        # Compute primal value
        ys[i] = f(arg_i...)
    end

    # Create output CoDual - the tangent will be modified by downstream pullbacks
    ys_codual = CoDual(ys, ys_tangent)

    function tmap_pullback!!(::NoRData)
        # For vectors, gradient comes from fdata (ys_tangent), not rdata
        # The downstream operation (e.g., sum) will have modified ys_tangent

        # Compute gradients for each element and accumulate into input tangents
        Δf = NoTangent()

        for i in 1:n
            arg_i = ntuple(j -> args[j][i], length(args))
            # Get cotangent for this element from output fdata
            δ_i = ys_tangent[i]

            # Use cache to compute pullback
            _, tangents_i = value_and_pullback!!(caches[i], δ_i, f, arg_i...)
            # tangents_i is (df, darg1, darg2, ...)

            Δf = _accum_tangents(Δf, tangents_i[1])

            # Accumulate into input tangents (fdata)
            for j in 1:length(args)
                args_tangents[j][i] += tangents_i[j + 1]
            end
        end

        # Return NoRData for all args since gradients are in fdata
        Δargs = ntuple(_ -> NoRData(), length(args))
        return NoTangent(), Δf, Δargs...
    end

    return ys_codual, tmap_pullback!!
end

"""
    rrule!! for SciMLBase.responsible_map

Implements reverse-mode AD for responsible_map by:
1. Forward pass: Compute primals and prepare pullback caches for each element
2. Reverse pass: Read gradients from output fdata, compute input gradients in reverse order (for stateful f)

Note: For vectors, Mooncake uses fdata (tangent field of CoDual) for gradients,
not rdata. The pullback receives NoRData() and must read the gradient from
the output's fdata which was modified by the downstream operation.
"""
function rrule!!(
        ::CoDual{typeof(SciMLBase.responsible_map)},
        f_dual::CoDual{F},
        args_dual::CoDual...
    ) where {F}
    # Extract primals and tangents (fdata)
    f = primal(f_dual)
    args = map(primal, args_dual)
    args_tangents = map(tangent, args_dual)

    n = length(args[1])

    # Compute first element to determine output type
    if n == 0
        # Empty case - infer type from function signature if possible
        T = Core.Compiler.return_type(f, Tuple{map(eltype, args)...})
        ys = Vector{T}(undef, 0)
        return zero_fcodual(ys), NoPullback(zero_fcodual(SciMLBase.responsible_map), f_dual, args_dual...)
    end

    # Forward pass: compute values and prepare caches
    caches = Vector{Any}(undef, n)

    # Compute first element to get the type
    arg_1 = ntuple(j -> args[j][1], length(args))
    caches[1] = prepare_pullback_cache(f, arg_1...)
    y1 = f(arg_1...)

    # Create properly typed output vector and its tangent (fdata)
    ys = Vector{typeof(y1)}(undef, n)
    ys[1] = y1
    ys_tangent = zeros(typeof(y1), n)

    # Compute remaining elements
    for i in 2:n
        arg_i = ntuple(j -> args[j][i], length(args))
        # Prepare cache for this call
        caches[i] = prepare_pullback_cache(f, arg_i...)
        # Compute primal value
        ys[i] = f(arg_i...)
    end

    # Create output CoDual - the tangent will be modified by downstream pullbacks
    ys_codual = CoDual(ys, ys_tangent)

    function responsible_map_pullback!!(::NoRData)
        # For vectors, gradient comes from fdata (ys_tangent), not rdata
        # The downstream operation (e.g., sum) will have modified ys_tangent

        # Compute gradients for each element and accumulate into input tangents
        # Apply pullbacks in reverse order for correctness with stateful f
        Δf = NoTangent()

        for i in n:-1:1
            arg_i = ntuple(j -> args[j][i], length(args))
            # Get cotangent for this element from output fdata
            δ_i = ys_tangent[i]

            # Use cache to compute pullback
            _, tangents_i = value_and_pullback!!(caches[i], δ_i, f, arg_i...)
            # tangents_i is (df, darg1, darg2, ...)

            Δf = _accum_tangents(Δf, tangents_i[1])

            # Accumulate into input tangents (fdata)
            for j in 1:length(args)
                args_tangents[j][i] += tangents_i[j + 1]
            end
        end

        # Return NoRData for all args since gradients are in fdata
        Δargs = ntuple(_ -> NoRData(), length(args))
        return NoTangent(), Δf, Δargs...
    end

    return ys_codual, responsible_map_pullback!!
end

end
