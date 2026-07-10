function enable_interpolation_sensitivitymode end

enable_interpolation_sensitivitymode(interp::Nothing) = nothing

# Pass through should be deprecated in the future, made for backwards compat
enable_interpolation_sensitivitymode(interp::AbstractDiffEqInterpolation) = interp

"""
$(TYPEDEF)

Marker type indicating that a solution's standard dense interpolation was disabled
because the solution was produced during sensitivity analysis.

Some non-AD sensitivity algorithms cannot safely differentiate through the
solver's native dense interpolation. Such solutions use `SensitivityInterpolation`
as their `interp` field so interpolation attempts can produce a targeted error
message instead of silently returning values from an unsupported interpolation
path. Save the needed time points directly with `saveat`, use `dense = false`
when linear or constant interpolation is sufficient, or choose a sensitivity
algorithm that differentiates through the solver when dense interpolation is
required.
"""
struct SensitivityInterpolation end

"""
$(TYPEDEF)

Third-order Hermite interpolation data for time-series SciML solutions.

`HermiteInterpolation` stores saved independent-variable values `t`, saved
solution values `u`, and saved derivatives `du`. It is callable through the
solution interpolation interface and supports derivative requests up to the
cubic polynomial's third derivative. Differential equation solvers use this when
they can provide derivative information at saved points and want solution calls
such as `sol(t)` to use Hermite reconstruction.

The `sensitivitymode` flag records whether the interpolation object has been
marked for sensitivity-aware behavior. Sensitivity handling may restrict which
interpolation paths are available; callers usually set this indirectly through
solver or sensitivity-algorithm options rather than constructing it manually.
"""
struct HermiteInterpolation{T1, T2, T3} <: AbstractDiffEqInterpolation
    t::T1
    u::T2
    du::T3
    sensitivitymode::Bool
end

function HermiteInterpolation(t, u, du; sensitivitymode = false)
    return HermiteInterpolation(t, u, du, sensitivitymode)
end

function enable_interpolation_sensitivitymode(interp::HermiteInterpolation)
    return HermiteInterpolation(interp.t, interp.u, interp.du, true)
end

"""
$(TYPEDEF)

First-order linear interpolation data for time-series SciML solutions.

`LinearInterpolation` stores saved independent-variable values `t` and saved
solution values `u`. It reconstructs values between adjacent saved points with
piecewise linear interpolation and supports first-derivative requests by
returning the segment slope where that operation is defined.

The `sensitivitymode` flag records whether sensitivity-aware behavior has been
enabled for the interpolation object. Linear interpolation is the standard
fallback when dense solver-specific interpolation is unavailable but saved values
can still be connected between time points.
"""
struct LinearInterpolation{T1, T2} <: AbstractDiffEqInterpolation
    t::T1
    u::T2
    sensitivitymode::Bool
end

function LinearInterpolation(t, u; sensitivitymode = false)
    return LinearInterpolation(t, u, sensitivitymode)
end

function enable_interpolation_sensitivitymode(interp::LinearInterpolation)
    return LinearInterpolation(interp.t, interp.u, true)
end

"""
$(TYPEDEF)

Piecewise constant interpolation data for time-series SciML solutions.

`ConstantInterpolation` stores saved independent-variable values `t` and saved
solution values `u`. It reconstructs values by holding the selected saved value
constant across each interval, with the interval side controlled by the
interpolation `continuity` argument. Derivative requests return zero for the
piecewise constant reconstruction where supported.

The `sensitivitymode` flag records whether sensitivity-aware behavior has been
enabled for the interpolation object. Constant interpolation is useful for
discrete-time states, zero-order-hold outputs, and solution objects whose saved
values should not be smoothed between time points.
"""
struct ConstantInterpolation{T1, T2} <: AbstractDiffEqInterpolation
    t::T1
    u::T2
    sensitivitymode::Bool
end

function ConstantInterpolation(t, u; sensitivitymode = false)
    return ConstantInterpolation(t, u, sensitivitymode)
end

function enable_interpolation_sensitivitymode(interp::ConstantInterpolation)
    return ConstantInterpolation(interp.t, interp.u, true)
end

"""
$(TYPEDEF)

Runtime-switched fallback interpolation for time-series SciML solutions.

`BasicInterpolation` is a single concrete type that covers both dense and
non-dense solutions, choosing the reconstruction at *runtime* through the
`dense` field rather than by the object's type: when `dense` is `true` it
reconstructs values with third-order Hermite interpolation (using the stored
derivatives `du`), and when `dense` is `false` it uses piecewise linear
interpolation. It stores saved independent-variable values `t`, saved solution
values `u`, and saved derivatives `du`, and it is callable through the standard
solution interpolation interface.

The point of this type is *type invariance*: the concrete type of a
`BasicInterpolation` does not depend on whether the solve was dense. A solver
wrapper that emits `BasicInterpolation` for both dense and non-dense solves
produces solutions whose concrete type is identical across those save settings,
so downstream code can hold a concretely-typed solution field and reassign it
across re-solves with different save arguments (for example checkpointing in
adjoint sensitivity analysis) without a type change.

Preserving that invariance requires that `du` always be the **same container
type** regardless of `dense`. When constructing a non-dense `BasicInterpolation`,
pass an empty container of the same type that a dense solve would use (e.g. an
empty `Vector` of the derivative element type) rather than `nothing`, so that
`typeof` is stable across dense and non-dense solutions of the same problem. The
non-dense path never reads `du`, so its contents are irrelevant when
`dense = false`; only its type matters.

The `sensitivitymode` flag records whether sensitivity-aware behavior has been
enabled; as with the other interpolation types it is normally set indirectly
through solver or sensitivity-algorithm options rather than manually.

The actual numerics are delegated to [`HermiteInterpolation`](@ref) and
[`LinearInterpolation`](@ref), so results are identical to using those types
directly in the corresponding mode.
"""
struct BasicInterpolation{tType, uType, duType} <: AbstractDiffEqInterpolation
    t::tType
    u::uType
    du::duType
    dense::Bool
    sensitivitymode::Bool
end

function BasicInterpolation(t, u, du, dense; sensitivitymode = false)
    return BasicInterpolation(t, u, du, dense, sensitivitymode)
end

function enable_interpolation_sensitivitymode(interp::BasicInterpolation)
    return BasicInterpolation(interp.t, interp.u, interp.du, interp.dense, true)
end

# Delegate the actual reconstruction to the Hermite/Linear interpolants so the
# numerics are identical to those types. The temporary interpolant only wraps
# references to the already-stored arrays (no data copy).
@inline function as_hermite_or_linear(id::BasicInterpolation)
    return id.dense ?
        HermiteInterpolation(id.t, id.u, id.du; sensitivitymode = id.sensitivitymode) :
        LinearInterpolation(id.t, id.u; sensitivitymode = id.sensitivitymode)
end

"""
    interp_summary(interp)

Return a short human-readable `String` describing the interpolation used by a solution,
e.g. `"3rd order Hermite"`. Accepts an [`AbstractDiffEqInterpolation`](@ref), an
[`AbstractSciMLSolution`](@ref) (in which case its `interp` field is summarized), or
`nothing` (no interpolation). Used when printing solutions.
"""
interp_summary(::AbstractDiffEqInterpolation) = "Unknown"
interp_summary(::HermiteInterpolation) = "3rd order Hermite"
interp_summary(::LinearInterpolation) = "1st order linear"
interp_summary(::ConstantInterpolation) = "Piecewise constant interpolation"
interp_summary(id::BasicInterpolation) = id.dense ? "3rd order Hermite" : "1st order linear"
interp_summary(::Nothing) = "No interpolation"
interp_summary(sol::AbstractSciMLSolution) = interp_summary(sol.interp)

const SENSITIVITY_INTERP_MESSAGE = """
Standard interpolation is disabled due to sensitivity analysis being
used for the gradients. Only linear and constant interpolations are
compatible with non-AD sensitivity analysis calculations. Either
utilize tooling like saveat to avoid post-solution interpolation, use
the keyword argument dense=false for linear or constant interpolations,
or use the keyword argument sensealg=SensitivityADPassThrough() to revert
to AD-based derivatives.
"""

function (id::HermiteInterpolation)(tvals, idxs, deriv, p, continuity::Symbol = :left)
    return interpolation(tvals, id, idxs, deriv, p, continuity)
end
function (id::HermiteInterpolation)(val, tvals, idxs, deriv, p, continuity::Symbol = :left)
    return interpolation!(val, tvals, id, idxs, deriv, p, continuity)
end
function (id::LinearInterpolation)(tvals, idxs, deriv, p, continuity::Symbol = :left)
    return interpolation(tvals, id, idxs, deriv, p, continuity)
end
function (id::LinearInterpolation)(val, tvals, idxs, deriv, p, continuity::Symbol = :left)
    return interpolation!(val, tvals, id, idxs, deriv, p, continuity)
end
function (id::ConstantInterpolation)(tvals, idxs, deriv, p, continuity::Symbol = :left)
    return interpolation(tvals, id, idxs, deriv, p, continuity)
end
function (id::ConstantInterpolation)(val, tvals, idxs, deriv, p, continuity::Symbol = :left)
    return interpolation!(val, tvals, id, idxs, deriv, p, continuity)
end
function (id::BasicInterpolation)(tvals, idxs, deriv, p, continuity::Symbol = :left)
    return as_hermite_or_linear(id)(tvals, idxs, deriv, p, continuity)
end
function (id::BasicInterpolation)(val, tvals, idxs, deriv, p, continuity::Symbol = :left)
    return as_hermite_or_linear(id)(val, tvals, idxs, deriv, p, continuity)
end

@inline function interpolation(
        tvals, id::I, idxs, deriv::D, p,
        continuity::Symbol = :left
    ) where {I, D}
    t = id.t
    u = id.u
    id isa HermiteInterpolation && (du = id.du)
    tdir = sign(t[end] - t[1])
    idx = sortperm(tvals, rev = tdir < 0)
    i = 2 # Start the search thinking it's between t[1] and t[2]
    t[end] == t[1] && (tvals[idx[1]] != t[1] || tvals[idx[end]] != t[1]) &&
        error("Solution interpolation cannot extrapolate from a single timepoint. Either solve on a longer timespan or use the local extrapolation from the integrator interface.")
    tdir * tvals[idx[end]] > tdir * t[end] &&
        error("Solution interpolation cannot extrapolate past the final timepoint. Either solve on a longer timespan or use the local extrapolation from the integrator interface.")
    tdir * tvals[idx[1]] < tdir * t[1] &&
        error("Solution interpolation cannot extrapolate before the first timepoint. Either start solving earlier or use the local extrapolation from the integrator interface.")
    if idxs isa Number
        vals = Vector{eltype(first(u))}(undef, length(tvals))
    elseif idxs isa AbstractVector
        vals = Vector{Vector{eltype(first(u))}}(undef, length(tvals))
    else
        vals = Vector{eltype(u)}(undef, length(tvals))
    end
    for j in idx
        tval = tvals[j]
        i = searchsortedfirst(@view(t[i:end]), tval, rev = tdir < 0) + i - 1 # It's in the interval t[i-1] to t[i]
        avoid_constant_ends = deriv != Val{0} #|| tval isa ForwardDiff.Dual
        avoid_constant_ends && i == 1 && (i += 1)
        if !avoid_constant_ends && t[i - 1] == tval # Can happen if it's the first value!
            if idxs === nothing
                vals[j] = u[i - 1]
            else
                vals[j] = u[i - 1][idxs]
            end
        elseif !avoid_constant_ends && t[i] == tval
            lasti = lastindex(t)
            k = continuity == :right && i + 1 <= lasti && t[i + 1] == tval ? i + 1 : i
            if idxs === nothing
                vals[j] = u[k]
            else
                vals[j] = u[k][idxs]
            end
        else
            id.sensitivitymode && error(SENSITIVITY_INTERP_MESSAGE)
            dt = t[i] - t[i - 1]
            Θ = (tval - t[i - 1]) / dt
            idxs_internal = idxs
            if id isa HermiteInterpolation
                vals[j] = interpolant(
                    Θ, id, dt, u[i - 1], u[i], du[i - 1], du[i],
                    idxs_internal, deriv
                )
            else
                vals[j] = interpolant(Θ, id, dt, u[i - 1], u[i], idxs_internal, deriv)
            end
        end
    end
    return DiffEqArray(vals, tvals)
end

"""
$(SIGNATURES)

Get the value at tvals where the solution is known at the
times t (sorted), with values u and derivatives ks
"""
@inline function interpolation!(
        vals, tvals, id::I, idxs, deriv::D, p,
        continuity::Symbol = :left
    ) where {I, D}
    t = id.t
    u = id.u
    id isa HermiteInterpolation && (du = id.du)
    tdir = sign(t[end] - t[1])
    idx = sortperm(tvals, rev = tdir < 0)
    i = 2 # Start the search thinking it's between t[1] and t[2]
    t[end] == t[1] && (tvals[idx[1]] != t[1] || tvals[idx[end]] != t[1]) &&
        error("Solution interpolation cannot extrapolate from a single timepoint. Either solve on a longer timespan or use the local extrapolation from the integrator interface.")
    tdir * tvals[idx[end]] > tdir * t[end] &&
        error("Solution interpolation cannot extrapolate past the final timepoint. Either solve on a longer timespan or use the local extrapolation from the integrator interface.")
    tdir * tvals[idx[1]] < tdir * t[1] &&
        error("Solution interpolation cannot extrapolate before the first timepoint. Either start solving earlier or use the local extrapolation from the integrator interface.")
    for j in idx
        tval = tvals[j]
        i = searchsortedfirst(@view(t[i:end]), tval, rev = tdir < 0) + i - 1 # It's in the interval t[i-1] to t[i]
        avoid_constant_ends = deriv != Val{0} #|| tval isa ForwardDiff.Dual
        avoid_constant_ends && i == 1 && (i += 1)
        if !avoid_constant_ends && t[i - 1] == tval # Can happen if it's the first value!
            if idxs === nothing
                vals[j] = u[i - 1]
            else
                vals[j] = u[i - 1][idxs]
            end
        elseif !avoid_constant_ends && t[i] == tval
            lasti = lastindex(t)
            k = continuity == :right && i + 1 <= lasti && t[i + 1] == tval ? i + 1 : i
            if idxs === nothing
                vals[j] = u[k]
            else
                vals[j] = u[k][idxs]
            end
        else
            id.sensitivitymode && error(SENSITIVITY_INTERP_MESSAGE)
            dt = t[i] - t[i - 1]
            Θ = (tval - t[i - 1]) / dt
            idxs_internal = idxs
            if eltype(u) <: Union{AbstractArray, ArrayPartition}
                if id isa HermiteInterpolation
                    interpolant!(
                        vals[j], Θ, id, dt, u[i - 1], u[i], du[i - 1], du[i],
                        idxs_internal, deriv
                    )
                else
                    interpolant!(vals[j], Θ, id, dt, u[i - 1], u[i], idxs_internal, deriv)
                end
            else
                if id isa HermiteInterpolation
                    vals[j] = interpolant(
                        Θ, id, dt, u[i - 1], u[i], du[i - 1], du[i],
                        idxs_internal, deriv
                    )
                else
                    vals[j] = interpolant(Θ, id, dt, u[i - 1], u[i], idxs_internal, deriv)
                end
            end
        end
    end
    return
end

"""
$(SIGNATURES)

Get the value at tval where the solution is known at the
times t (sorted), with values u and derivatives ks
"""
@inline function interpolation(
        tval::Number, id::I, idxs, deriv::D, p,
        continuity::Symbol = :left
    ) where {I, D}
    t = id.t
    u = id.u
    id isa HermiteInterpolation && (du = id.du)
    tdir = sign(t[end] - t[1])
    t[end] == t[1] && tval != t[end] &&
        error("Solution interpolation cannot extrapolate from a single timepoint. Either solve on a longer timespan or use the local extrapolation from the integrator interface.")
    tdir * tval > tdir * t[end] &&
        error("Solution interpolation cannot extrapolate past the final timepoint. Either solve on a longer timespan or use the local extrapolation from the integrator interface.")
    tdir * tval < tdir * t[1] &&
        error("Solution interpolation cannot extrapolate before the first timepoint. Either start solving earlier or use the local extrapolation from the integrator interface.")
    @inbounds i = searchsortedfirst(t, tval, rev = tdir < 0) # It's in the interval t[i-1] to t[i]
    avoid_constant_ends = deriv != Val{0} #|| tval isa ForwardDiff.Dual
    avoid_constant_ends && i == 1 && (i += 1)
    if !avoid_constant_ends && t[i] == tval
        lasti = lastindex(t)
        k = continuity == :right && i + 1 <= lasti && t[i + 1] == tval ? i + 1 : i
        if idxs === nothing
            val = u[k]
        else
            val = u[k][idxs]
        end
    elseif !avoid_constant_ends && t[i - 1] == tval # Can happen if it's the first value!
        if idxs === nothing
            val = u[i - 1]
        else
            val = u[i - 1][idxs]
        end
    else
        id.sensitivitymode && error(SENSITIVITY_INTERP_MESSAGE)
        dt = t[i] - t[i - 1]
        Θ = (tval - t[i - 1]) / dt
        idxs_internal = idxs
        if id isa HermiteInterpolation
            val = interpolant(
                Θ, id, dt, u[i - 1], u[i], du[i - 1], du[i], idxs_internal,
                deriv
            )
        else
            val = interpolant(Θ, id, dt, u[i - 1], u[i], idxs_internal, deriv)
        end
    end
    return val
end

"""
$(SIGNATURES)

Get the value at tval where the solution is known at the
times t (sorted), with values u and derivatives ks
"""
@inline function interpolation!(
        out, tval::Number, id::I, idxs, deriv::D, p,
        continuity::Symbol = :left
    ) where {I, D}
    t = id.t
    u = id.u
    id isa HermiteInterpolation && (du = id.du)
    tdir = sign(t[end] - t[1])
    t[end] == t[1] && tval != t[end] &&
        error("Solution interpolation cannot extrapolate from a single timepoint. Either solve on a longer timespan or use the local extrapolation from the integrator interface.")
    tdir * tval > tdir * t[end] &&
        error("Solution interpolation cannot extrapolate past the final timepoint. Either solve on a longer timespan or use the local extrapolation from the integrator interface.")
    tdir * tval < tdir * t[1] &&
        error("Solution interpolation cannot extrapolate before the first timepoint. Either start solving earlier or use the local extrapolation from the integrator interface.")
    @inbounds i = searchsortedfirst(t, tval, rev = tdir < 0) # It's in the interval t[i-1] to t[i]
    avoid_constant_ends = deriv != Val{0} #|| tval isa ForwardDiff.Dual
    avoid_constant_ends && i == 1 && (i += 1)
    return if !avoid_constant_ends && t[i] == tval
        lasti = lastindex(t)
        k = continuity == :right && i + 1 <= lasti && t[i + 1] == tval ? i + 1 : i
        if idxs === nothing
            copy!(out, u[k])
        else
            copy!(out, u[k][idxs])
        end
    elseif !avoid_constant_ends && t[i - 1] == tval # Can happen if it's the first value!
        if idxs === nothing
            copy!(out, u[i - 1])
        else
            copy!(out, u[i - 1][idxs])
        end
    else
        id.sensitivitymode && error(SENSITIVITY_INTERP_MESSAGE)
        dt = t[i] - t[i - 1]
        Θ = (tval - t[i - 1]) / dt
        idxs_internal = idxs
        if id isa HermiteInterpolation
            interpolant!(
                out, Θ, id, dt, u[i - 1], u[i], du[i - 1], du[i], idxs_internal,
                deriv
            )
        else
            interpolant!(out, Θ, id, dt, u[i - 1], u[i], idxs_internal, deriv)
        end
    end
end

@inline function interpolant(
        Θ, id::AbstractDiffEqInterpolation, dt, y₀, y₁, dy₀, dy₁, idxs,
        ::Type{Val{D}}
    ) where {D}
    error("$(string(typeof(id))) for $(D)th order not implemented")
end
##################### Hermite Interpolants

"""
Hairer Norsett Wanner Solving Ordinary Differential Equations I - Nonstiff Problems Page 190

Hermite Interpolation
"""
@inline function interpolant(
        Θ, id::HermiteInterpolation, dt, y₀, y₁, dy₀, dy₁, idxs,
        T::Type{Val{0}}
    )
    if idxs === nothing
        out = @. (1 - Θ) * y₀ + Θ * y₁ +
            Θ * (Θ - 1) * ((1 - 2Θ) * (y₁ - y₀) + (Θ - 1) * dt * dy₀ + Θ * dt * dy₁)
    elseif idxs isa Number
        out = (1 - Θ) * y₀[idxs] + Θ * y₁[idxs] +
            Θ * (Θ - 1) *
            (
            (1 - 2Θ) * (y₁[idxs] - y₀[idxs]) +
                (Θ - 1) * dt * dy₀[idxs] + Θ * dt * dy₁[idxs]
        )
    else
        out = similar(y₀, axes(idxs))
        @views @. out = (1 - Θ) * y₀[idxs] + Θ * y₁[idxs] +
            Θ * (Θ - 1) *
            (
            (1 - 2Θ) * (y₁[idxs] - y₀[idxs]) +
                (Θ - 1) * dt * dy₀[idxs] + Θ * dt * dy₁[idxs]
        )
    end
    return out
end

"""
Hermite Interpolation
"""
@inline function interpolant(
        Θ, id::HermiteInterpolation, dt, y₀, y₁, dy₀, dy₁, idxs,
        T::Type{Val{1}}
    )
    if idxs === nothing
        out = @. dy₀ +
            Θ * (
            -4 * dt * dy₀ - 2 * dt * dy₁ - 6 * y₀ +
                Θ * (3 * dt * dy₀ + 3 * dt * dy₁ + 6 * y₀ - 6 * y₁) + 6 * y₁
        ) / dt
    elseif idxs isa Number
        out = dy₀[idxs] +
            Θ * (
            -4 * dt * dy₀[idxs] -
                2 * dt * dy₁[idxs] - 6 * y₀[idxs] +
                Θ * (
                3 * dt * dy₀[idxs] + 3 * dt * dy₁[idxs] +
                    6 * y₀[idxs] - 6 * y₁[idxs]
            ) + 6 * y₁[idxs]
        ) / dt
    else
        out = similar(y₀, axes(idxs))
        @views @. out = dy₀[idxs] +
            Θ * (
            -4 * dt * dy₀[idxs] -
                2 * dt * dy₁[idxs] - 6 * y₀[idxs] +
                Θ * (
                3 * dt * dy₀[idxs] + 3 * dt * dy₁[idxs] +
                    6 * y₀[idxs] - 6 * y₁[idxs]
            ) + 6 * y₁[idxs]
        ) / dt
    end
    return out
end

"""
Hermite Interpolation
"""
@inline function interpolant(
        Θ, id::HermiteInterpolation, dt, y₀, y₁, dy₀, dy₁, idxs,
        T::Type{Val{2}}
    )
    if idxs === nothing
        out = @. (
            -4 * dt * dy₀ - 2 * dt * dy₁ - 6 * y₀ +
                Θ * (6 * dt * dy₀ + 6 * dt * dy₁ + 12 * y₀ - 12 * y₁) + 6 * y₁
        ) /
            (dt * dt)
    elseif idxs isa Number
        out = (
            -4 * dt * dy₀[idxs] - 2 * dt * dy₁[idxs] - 6 * y₀[idxs] +
                Θ * (
                6 * dt * dy₀[idxs] + 6 * dt * dy₁[idxs] + 12 * y₀[idxs] -
                    12 * y₁[idxs]
            ) + 6 * y₁[idxs]
        ) / (dt * dt)
    else
        out = similar(y₀, axes(idxs))
        @views @. out = (
            -4 * dt * dy₀[idxs] - 2 * dt * dy₁[idxs] - 6 * y₀[idxs] +
                Θ * (
                6 * dt * dy₀[idxs] + 6 * dt * dy₁[idxs] + 12 * y₀[idxs] -
                    12 * y₁[idxs]
            ) + 6 * y₁[idxs]
        ) / (dt * dt)
    end
    return out
end

"""
Hermite Interpolation
"""
@inline function interpolant(
        Θ, id::HermiteInterpolation, dt, y₀, y₁, dy₀, dy₁, idxs,
        T::Type{Val{3}}
    )
    if idxs === nothing
        out = @. (6 * dt * dy₀ + 6 * dt * dy₁ + 12 * y₀ - 12 * y₁) / (dt * dt * dt)
    elseif idxs isa Number
        out = (
            6 * dt * dy₀[idxs] + 6 * dt * dy₁[idxs] +
                12 * y₀[idxs] - 12 * y₁[idxs]
        ) / (dt * dt * dt)
    else
        out = similar(y₀, axes(idxs))
        @views @. out = (
            6 * dt * dy₀[idxs] + 6 * dt * dy₁[idxs] +
                12 * y₀[idxs] - 12 * y₁[idxs]
        ) / (dt * dt * dt)
    end
    return out
end

"""
Hairer Norsett Wanner Solving Ordinary Differential Euations I - Nonstiff Problems Page 190

Hermite Interpolation
"""
@inline function interpolant!(
        out, Θ, id::HermiteInterpolation, dt, y₀, y₁, dy₀, dy₁, idxs,
        T::Type{Val{0}}
    )
    if out === nothing
        return (1 - Θ) * y₀[idxs] + Θ * y₁[idxs] +
            Θ * (Θ - 1) *
            (
            (1 - 2Θ) * (y₁[idxs] - y₀[idxs]) + (Θ - 1) * dt * dy₀[idxs] +
                Θ * dt * dy₁[idxs]
        )
    elseif idxs === nothing
        @. out = (1 - Θ) * y₀ + Θ * y₁ +
            Θ * (Θ - 1) * ((1 - 2Θ) * (y₁ - y₀) + (Θ - 1) * dt * dy₀ + Θ * dt * dy₁)
    else
        @views @. out = (1 - Θ) * y₀[idxs] + Θ * y₁[idxs] +
            Θ * (Θ - 1) *
            (
            (1 - 2Θ) * (y₁[idxs] - y₀[idxs]) + (Θ - 1) * dt * dy₀[idxs] +
                Θ * dt * dy₁[idxs]
        )
    end
end

"""
Hermite Interpolation
"""
@inline function interpolant!(
        out, Θ, id::HermiteInterpolation, dt, y₀, y₁, dy₀, dy₁, idxs,
        T::Type{Val{1}}
    )
    if out === nothing
        return dy₀[idxs] +
            Θ * (
            -4 * dt * dy₀[idxs] - 2 * dt * dy₁[idxs] - 6 * y₀[idxs] +
                Θ *
                (3 * dt * dy₀[idxs] + 3 * dt * dy₁[idxs] + 6 * y₀[idxs] - 6 * y₁[idxs]) +
                6 * y₁[idxs]
        ) / dt
    elseif idxs === nothing
        @. out = dy₀ +
            Θ * (
            -4 * dt * dy₀ - 2 * dt * dy₁ - 6 * y₀ +
                Θ * (3 * dt * dy₀ + 3 * dt * dy₁ + 6 * y₀ - 6 * y₁) + 6 * y₁
        ) / dt
    else
        @views @. out = dy₀[idxs] +
            Θ * (
            -4 * dt * dy₀[idxs] - 2 * dt * dy₁[idxs] - 6 * y₀[idxs] +
                Θ * (
                3 * dt * dy₀[idxs] + 3 * dt * dy₁[idxs] + 6 * y₀[idxs] -
                    6 * y₁[idxs]
            ) + 6 * y₁[idxs]
        ) / dt
    end
end

"""
Hermite Interpolation
"""
@inline function interpolant!(
        out, Θ, id::HermiteInterpolation, dt, y₀, y₁, dy₀, dy₁, idxs,
        T::Type{Val{2}}
    )
    if out === nothing
        return (
            -4 * dt * dy₀[idxs] - 2 * dt * dy₁[idxs] - 6 * y₀[idxs] +
                Θ *
                (6 * dt * dy₀[idxs] + 6 * dt * dy₁[idxs] + 12 * y₀[idxs] - 12 * y₁[idxs]) +
                6 * y₁[idxs]
        ) / (dt * dt)
    elseif idxs === nothing
        @. out = (
            -4 * dt * dy₀ - 2 * dt * dy₁ - 6 * y₀ +
                Θ * (6 * dt * dy₀ + 6 * dt * dy₁ + 12 * y₀ - 12 * y₁) + 6 * y₁
        ) /
            (dt * dt)
    else
        @views @. out = (
            -4 * dt * dy₀[idxs] - 2 * dt * dy₁[idxs] - 6 * y₀[idxs] +
                Θ * (
                6 * dt * dy₀[idxs] + 6 * dt * dy₁[idxs] + 12 * y₀[idxs] -
                    12 * y₁[idxs]
            ) + 6 * y₁[idxs]
        ) / (dt * dt)
    end
end

"""
Hermite Interpolation
"""
@inline function interpolant!(
        out, Θ, id::HermiteInterpolation, dt, y₀, y₁, dy₀, dy₁, idxs,
        T::Type{Val{3}}
    )
    if out === nothing
        return (6 * dt * dy₀[idxs] + 6 * dt * dy₁[idxs] + 12 * y₀[idxs] - 12 * y₁[idxs]) /
            (dt * dt * dt)
    elseif idxs === nothing
        @. out = (6 * dt * dy₀ + 6 * dt * dy₁ + 12 * y₀ - 12 * y₁) / (dt * dt * dt)
    else
        @views @. out = (
            6 * dt * dy₀[idxs] + 6 * dt * dy₁[idxs] + 12 * y₀[idxs] -
                12 * y₁[idxs]
        ) / (dt * dt * dt)
    end
end

############################### Linear Interpolants

"""
Linear Interpolation
"""
@inline function interpolant(Θ, id::LinearInterpolation, dt, y₀, y₁, idxs, T::Type{Val{0}})
    Θm1 = (1 - Θ)
    if idxs === nothing
        out = @. Θm1 * y₀ + Θ * y₁
    elseif idxs isa Number
        out = Θm1 * y₀[idxs] + Θ * y₁[idxs]
    else
        out = similar(y₀, axes(idxs))
        @views @. out = Θm1 * y₀[idxs] + Θ * y₁[idxs]
    end
    return out
end

@inline function interpolant(Θ, id::LinearInterpolation, dt, y₀, y₁, idxs, T::Type{Val{1}})
    if idxs === nothing
        out = @. (y₁ - y₀) / dt
    elseif idxs isa Number
        out = (y₁[idxs] - y₀[idxs]) / dt
    else
        out = similar(y₀, axes(idxs))
        @views @. out = (y₁[idxs] - y₀[idxs]) / dt
    end
    return out
end

"""
Linear Interpolation
"""
@inline function interpolant!(
        out, Θ, id::LinearInterpolation, dt, y₀, y₁, idxs,
        T::Type{Val{0}}
    )
    Θm1 = (1 - Θ)
    if out === nothing
        return Θm1 * y₀[idxs] + Θ * y₁[idxs]
    elseif idxs === nothing
        @. out = Θm1 * y₀ + Θ * y₁
    else
        @views @. out = Θm1 * y₀[idxs] + Θ * y₁[idxs]
    end
end

"""
Linear Interpolation
"""
@inline function interpolant!(
        out, Θ, id::LinearInterpolation, dt, y₀, y₁, idxs,
        T::Type{Val{1}}
    )
    if out === nothing
        return (y₁[idxs] - y₀[idxs]) / dt
    elseif idxs === nothing
        @. out = (y₁ - y₀) / dt
    else
        @views @. out = (y₁[idxs] - y₀[idxs]) / dt
    end
end

############################### Linear Interpolants

"""
Constant Interpolation
"""
@inline function interpolant(
        Θ, id::ConstantInterpolation, dt, y₀, y₁, idxs,
        T::Type{Val{0}}
    )
    if idxs === nothing
        out = @. y₀
    elseif idxs isa Number
        out = y₀[idxs]
    else
        out = similar(y₀, axes(idxs))
        @views @. out = y₀[idxs]
    end
    return out
end

@inline function interpolant(
        Θ, id::ConstantInterpolation, dt, y₀, y₁, idxs,
        T::Type{Val{1}}
    )
    if idxs === nothing
        out = zeros(eltype(y₀), length(y₀))
    elseif idxs isa Number
        out = zero(eltype(y₀))
    else
        out = similar(y₀, axes(idxs))
        @views @. out = 0
    end
    return out
end

"""
Constant Interpolation
"""
@inline function interpolant!(
        out, Θ, id::ConstantInterpolation, dt, y₀, y₁, idxs,
        T::Type{Val{0}}
    )
    if out === nothing
        return y₀[idxs]
    elseif idxs === nothing
        @. out = y₀
    else
        @views @. out = y₀[idxs]
    end
end

"""
Constant Interpolation
"""
@inline function interpolant!(
        out, Θ, id::ConstantInterpolation, dt, y₀, y₁, idxs,
        T::Type{Val{1}}
    )
    if out === nothing
        return zeros(eltype(y₀), length(idxs))
    else
        @. out = 0
    end
end

"""
    strip_interpolation(id::AbstractDiffEqInterpolation)

Returns a copy of the interpolation stripped of its function, to accommodate serialization.
If the interpolation object has no function, returns the interpolation object as is.
"""
strip_interpolation(id::AbstractDiffEqInterpolation) = id
strip_interpolation(id::HermiteInterpolation) = id
strip_interpolation(id::LinearInterpolation) = id
strip_interpolation(id::ConstantInterpolation) = id
strip_interpolation(id::BasicInterpolation) = id
