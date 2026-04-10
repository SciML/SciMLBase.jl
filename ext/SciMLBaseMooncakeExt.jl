module SciMLBaseMooncakeExt

using SciMLBase, Mooncake
using SciMLBase: ADOriginator, ChainRulesOriginator, MooncakeOriginator,
    AbstractTimeseriesSolution, getobserved
using SymbolicIndexingInterface: symbolic_type, NotSymbolic, variable_index,
    parameter_values
import Mooncake: rrule!!, CoDual, zero_fcodual, @is_primitive,
    @from_rrule, @zero_adjoint, @mooncake_overlay, MinimalCtx,
    NoPullback, NoFData, NoRData, NoTangent, fdata, zero_tangent,
    primal, tangent, build_rrule, prepare_pullback_cache, value_and_pullback!!

# OverrideInitData and ODENLStepData are solver/initialization infrastructure
# embedded in ODEFunction type parameters. They are not differentiable, but their
# deeply nested type parameters (NonlinearProblem, RuntimeGeneratedFunction,
# InitializationMetadata, etc.) cause Mooncake's abstract interpretation to hang
# during type inference. Marking them NoTangent avoids generating tangent code
# for these fields entirely.
Mooncake.tangent_type(::Type{<:SciMLBase.OverrideInitData}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{<:SciMLBase.ODENLStepData}) = Mooncake.NoTangent

# Non-differentiable helpers — these use runtime reflection (`methods`) or
# only validate/dispatch and have no numerical gradient contribution.
@zero_adjoint MinimalCtx Tuple{typeof(SciMLBase.numargs), Any}
@zero_adjoint MinimalCtx Tuple{typeof(SciMLBase.checkkwargs), Any}
@zero_adjoint MinimalCtx Tuple{typeof(SciMLBase.isinplace), Any, Any}
@zero_adjoint MinimalCtx Tuple{typeof(SciMLBase.isinplace), Any, Any, Any}
@zero_adjoint MinimalCtx Tuple{typeof(SciMLBase.isinplace), Any, Any, Any, Any}

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

# =============================================================================
# Symbolic indexing of solutions: getindex(sol, sym)
# =============================================================================
#
# Mooncake cannot differentiate through `sol[sym]` for symbolic `sym` because
# the SymbolicIndexingInterface dispatch chain uses hash-consed symbols,
# task-local values, IdDicts, and atomic pointers — none of which are
# differentiable infrastructure. Make `getindex(::AbstractTimeseriesSolution,
# sym)` a primitive that bypasses the dispatch and:
#
# - For state variables: scatter the gradient directly into `sol.u[k][i]`
# - For observables: build a Mooncake derived rule for the observed function
#   and apply it (analogous to how `SciMLBaseChainRulesCoreExt` uses
#   `rrule_via_ad` to recurse into Zygote for the observed function).

@is_primitive MinimalCtx Tuple{typeof(getindex), AbstractTimeseriesSolution, Any}
@is_primitive MinimalCtx Tuple{
    typeof(getindex), AbstractTimeseriesSolution, Any, Integer,
}

# Differentiate the observed function `getter(sym, u, p, t)` once with
# Mooncake. The pullback `pb(dy)` mutates the fdata captured in the input
# CoDuals so the gradients land in the locations we control: `u_fd` for the
# state and `p_fd` for the parameters. This is the Mooncake equivalent of
# how SciMLBaseChainRulesCoreExt uses `rrule_via_ad` to recurse into Zygote.
function _run_observable_pullback(getter, s, u_jp, p, t_jp, dy)
    sig = Tuple{typeof(getter), typeof(s), typeof(u_jp), typeof(p), typeof(t_jp)}
    rule = build_rrule(sig)

    # Allocate fresh fdata for u and p so the gradient lands in known
    # buffers rather than the (possibly aliased) original tangents.
    u_fd = zero(u_jp)
    p_fd = fdata(zero_tangent(p))
    getter_cd = CoDual(getter, fdata(zero_tangent(getter)))
    sym_cd = CoDual(s, fdata(zero_tangent(s)))
    u_cd = CoDual(u_jp, u_fd)
    p_cd = CoDual(p, p_fd)
    t_cd = CoDual(t_jp, fdata(zero_tangent(t_jp)))

    _, pb = rule(getter_cd, sym_cd, u_cd, p_cd, t_cd)
    pb(dy)
    return u_fd, p_fd
end

# Accumulate parameter gradient `p_fd` (an FData) into `dest_p_tangent`
# (which may be a Tangent for immutable structs or an FData/MutableTangent
# for other layouts). We can't use Mooncake.increment!! because of type
# mismatches between Tangent and FData. Instead, walk the NamedTuple
# structure and accumulate Vector-typed leaves in place.
function _accumulate_p_fdata!(dest, src)
    (dest isa NoFData || dest isa NoTangent || src isa NoFData || src isa NoTangent) &&
        return nothing
    if dest isa Vector{<:Real} && src isa Vector{<:Real}
        @. dest += src
        return nothing
    end
    # Tangent / MutableTangent use .fields; FData uses .data.
    dest_fields = if dest isa Mooncake.FData
        dest.data
    elseif dest isa Mooncake.Tangent || dest isa Mooncake.MutableTangent
        dest.fields
    else
        return nothing
    end
    src_fields = if src isa Mooncake.FData
        src.data
    elseif src isa Mooncake.Tangent || src isa Mooncake.MutableTangent
        src.fields
    else
        return nothing
    end
    for name in propertynames(dest_fields)
        hasproperty(src_fields, name) || continue
        _accumulate_p_fdata!(getfield(dest_fields, name), getfield(src_fields, name))
    end
    return nothing
end

function _observable_pullback_at!(sol_fdata, VA, s, dy, jp)
    getter = getobserved(VA)
    u_jp = VA.u[jp]
    p = parameter_values(VA.prob)
    t_jp = VA.t[jp]

    u_fd, p_fd = _run_observable_pullback(getter, s, u_jp, p, t_jp, dy)

    # Scatter u-grad into sol_fdata.u at index jp
    u_dest = sol_fdata.data.u[jp]
    @. u_dest += u_fd

    # Propagate p-grad into sol_fdata.prob.p (if accessible).
    prob_fd = sol_fdata.data.prob
    if prob_fd isa Mooncake.MutableTangent && hasfield(typeof(prob_fd.fields), :p)
        _accumulate_p_fdata!(prob_fd.fields.p, p_fd)
    end
    return nothing
end

function rrule!!(
        ::CoDual{typeof(getindex)},
        sol::CoDual{<:AbstractTimeseriesSolution},
        sym::CoDual,
    )
    VA = sol.x
    s = sym.x
    y = VA[s]
    # Allocate a zero-initialized fdata for the output. Downstream rrules
    # (e.g. `sum`) mutate this in-place to accumulate the output gradient.
    y_fdata = zero(y)
    sol_fdata = sol.dx

    function _scatter_pullback(::NoRData)
        i = symbolic_type(s) != NotSymbolic() ? variable_index(VA, s) : s
        if i !== nothing
            # State variable: scatter y_fdata[k] into sol_fdata.u[k][i].
            u_fd = sol_fdata.data.u
            for k in eachindex(VA.u)
                u_fd[k][i] += y_fdata[k]
            end
        else
            # Observable: differentiate the observed function via a Mooncake
            # derived rule, once per saved time point.
            for k in eachindex(VA.u)
                _observable_pullback_at!(sol_fdata, VA, s, y_fdata[k], k)
            end
        end
        return (NoRData(), NoRData(), NoRData())
    end

    return CoDual(y, y_fdata), _scatter_pullback
end

function rrule!!(
        ::CoDual{typeof(getindex)},
        sol::CoDual{<:AbstractTimeseriesSolution},
        sym::CoDual,
        j::CoDual{<:Integer},
    )
    VA = sol.x
    s = sym.x
    jp = j.x
    y = VA[s, jp]
    sol_fdata = sol.dx

    # Scalar output: gradient comes via dy (rdata).
    function _scatter_pullback_indexed(dy)
        i = symbolic_type(s) != NotSymbolic() ? variable_index(VA, s) : s
        if i !== nothing
            sol_fdata.data.u[jp][i] += dy
        else
            _observable_pullback_at!(sol_fdata, VA, s, dy, jp)
        end
        return (NoRData(), NoRData(), NoRData(), NoRData())
    end

    return zero_fcodual(y), _scatter_pullback_indexed
end

# NOTE: The ChainRules extension also defines rrules for constructors
# (SDEProblem, ODESolution, RODESolution, IntervalNonlinearProblem,
# EnsembleSolution) and for `getproperty(::NonlinearProblem, ::Symbol)`.
# These are Zygote-specific workarounds for how Zygote handles mutable
# struct field accumulation and constructor tracing. Mooncake's MutableTangent
# handles struct construction and field access with proper cache reset
# semantics natively, so those rrules are not needed on the Mooncake path.

end
