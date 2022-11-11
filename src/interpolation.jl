"""
$(TYPEDEF)
"""
struct HermiteInterpolation{T1, T2, T3} <: AbstractDiffEqInterpolation
    t::T1
    u::T2
    du::T3
end

"""
$(TYPEDEF)
"""
struct LinearInterpolation{T1, T2} <: AbstractDiffEqInterpolation
    t::T1
    u::T2
end

"""
$(TYPEDEF)
"""
struct ConstantInterpolation{T1, T2} <: AbstractDiffEqInterpolation
    t::T1
    u::T2
end

"""
$(TYPEDEF)
"""
struct SensitivityInterpolation{T1, T2} <: AbstractDiffEqInterpolation
    t::T1
    u::T2
end

interp_summary(::AbstractDiffEqInterpolation) = "Unknown"
interp_summary(::HermiteInterpolation) = "3rd order Hermite"
interp_summary(::LinearInterpolation) = "1st order linear"
interp_summary(::ConstantInterpolation) = "Piecewise constant interpolation"
interp_summary(::Nothing) = "No interpolation"
function interp_summary(::SensitivityInterpolation)
    "Interpolation disabled due to sensitivity analysis"
end
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
    interpolation(tvals, id, idxs, deriv, p, continuity)
end
function (id::HermiteInterpolation)(val, tvals, idxs, deriv, p, continuity::Symbol = :left)
    interpolation!(val, tvals, id, idxs, deriv, p, continuity)
end
function (id::LinearInterpolation)(tvals, idxs, deriv, p, continuity::Symbol = :left)
    interpolation(tvals, id, idxs, deriv, p, continuity)
end
function (id::LinearInterpolation)(val, tvals, idxs, deriv, p, continuity::Symbol = :left)
    interpolation!(val, tvals, id, idxs, deriv, p, continuity)
end
function (id::ConstantInterpolation)(tvals, idxs, deriv, p, continuity::Symbol = :left)
    interpolation(tvals, id, idxs, deriv, p, continuity)
end
function (id::ConstantInterpolation)(val, tvals, idxs, deriv, p, continuity::Symbol = :left)
    interpolation!(val, tvals, id, idxs, deriv, p, continuity)
end
function (id::SensitivityInterpolation)(tvals, idxs, deriv, p, continuity::Symbol = :left)
    interpolation(tvals, id, idxs, deriv, p, continuity)
end
function (id::SensitivityInterpolation)(val, tvals, idxs, deriv, p,
                                        continuity::Symbol = :left)
    interpolation!(val, tvals, id, idxs, deriv, p, continuity)
end

@inline function interpolation(tvals, id::I, idxs, deriv::D, p,
                               continuity::Symbol = :left) where {I, D}
    t = id.t
    u = id.u
    typeof(id) <: HermiteInterpolation && (du = id.du)
    tdir = sign(t[end] - t[1])
    idx = sortperm(tvals, rev = tdir < 0)
    i = 2 # Start the search thinking it's between t[1] and t[2]
    tdir * tvals[idx[end]] > tdir * t[end] &&
        error("Solution interpolation cannot extrapolate past the final timepoint. Either solve on a longer timespan or use the local extrapolation from the integrator interface.")
    tdir * tvals[idx[1]] < tdir * t[1] &&
        error("Solution interpolation cannot extrapolate before the first timepoint. Either start solving earlier or use the local extrapolation from the integrator interface.")
    if typeof(idxs) <: Number
        vals = Vector{eltype(first(u))}(undef, length(tvals))
    elseif typeof(idxs) <: AbstractVector
        vals = Vector{Vector{eltype(first(u))}}(undef, length(tvals))
    else
        vals = Vector{eltype(u)}(undef, length(tvals))
    end
    @inbounds for j in idx
        tval = tvals[j]
        i = searchsortedfirst(@view(t[i:end]), tval, rev = tdir < 0) + i - 1 # It's in the interval t[i-1] to t[i]
        avoid_constant_ends = deriv != Val{0} #|| typeof(tval) <: ForwardDiff.Dual
        avoid_constant_ends && i == 1 && (i += 1)
        if !avoid_constant_ends && t[i] == tval
            lasti = lastindex(t)
            k = continuity == :right && i + 1 <= lasti && t[i + 1] == tval ? i + 1 : i
            if idxs === nothing
                vals[j] = u[k]
            else
                vals[j] = u[k][idxs]
            end
        elseif !avoid_constant_ends && t[i - 1] == tval # Can happen if it's the first value!
            if idxs === nothing
                vals[j] = u[i - 1]
            else
                vals[j] = u[i - 1][idxs]
            end
        else
            typeof(id) <: SensitivityInterpolation && error(SENSITIVITY_INTERP_MESSAGE)
            dt = t[i] - t[i - 1]
            Θ = (tval - t[i - 1]) / dt
            idxs_internal = idxs
            if typeof(id) <: HermiteInterpolation
                vals[j] = interpolant(Θ, id, dt, u[i - 1], u[i], du[i - 1], du[i],
                                      idxs_internal, deriv)
            else
                vals[j] = interpolant(Θ, id, dt, u[i - 1], u[i], idxs_internal, deriv)
            end
        end
    end
    DiffEqArray(vals, tvals)
end

"""
$(SIGNATURES)

Get the value at tvals where the solution is known at the
times t (sorted), with values u and derivatives ks
"""
@inline function interpolation!(vals, tvals, id::I, idxs, deriv::D, p,
                                continuity::Symbol = :left) where {I, D}
    t = id.t
    u = id.u
    typeof(id) <: HermiteInterpolation && (du = id.du)
    tdir = sign(t[end] - t[1])
    idx = sortperm(tvals, rev = tdir < 0)
    i = 2 # Start the search thinking it's between t[1] and t[2]
    tdir * tvals[idx[end]] > tdir * t[end] &&
        error("Solution interpolation cannot extrapolate past the final timepoint. Either solve on a longer timespan or use the local extrapolation from the integrator interface.")
    tdir * tvals[idx[1]] < tdir * t[1] &&
        error("Solution interpolation cannot extrapolate before the first timepoint. Either start solving earlier or use the local extrapolation from the integrator interface.")
    @inbounds for j in idx
        tval = tvals[j]
        i = searchsortedfirst(@view(t[i:end]), tval, rev = tdir < 0) + i - 1 # It's in the interval t[i-1] to t[i]
        avoid_constant_ends = deriv != Val{0} #|| typeof(tval) <: ForwardDiff.Dual
        avoid_constant_ends && i == 1 && (i += 1)
        if !avoid_constant_ends && t[i] == tval
            lasti = lastindex(t)
            k = continuity == :right && i + 1 <= lasti && t[i + 1] == tval ? i + 1 : i
            if idxs === nothing
                vals[j] = u[k]
            else
                vals[j] = u[k][idxs]
            end
        elseif !avoid_constant_ends && t[i - 1] == tval # Can happen if it's the first value!
            if idxs === nothing
                vals[j] = u[i - 1]
            else
                vals[j] = u[i - 1][idxs]
            end
        else
            typeof(id) <: SensitivityInterpolation && error(SENSITIVITY_INTERP_MESSAGE)
            dt = t[i] - t[i - 1]
            Θ = (tval - t[i - 1]) / dt
            idxs_internal = idxs
            if eltype(u) <: Union{AbstractArray, ArrayPartition}
                if typeof(id) <: HermiteInterpolation
                    interpolant!(vals[j], Θ, id, dt, u[i - 1], u[i], du[i - 1], du[i],
                                 idxs_internal, deriv)
                else
                    interpolant!(vals[j], Θ, id, dt, u[i - 1], u[i], idxs_internal, deriv)
                end
            else
                if typeof(id) <: HermiteInterpolation
                    vals[j] = interpolant(Θ, id, dt, u[i - 1], u[i], du[i - 1], du[i],
                                          idxs_internal, deriv)
                else
                    vals[j] = interpolant(Θ, id, dt, u[i - 1], u[i], idxs_internal, deriv)
                end
            end
        end
    end
end

"""
$(SIGNATURES)

Get the value at tval where the solution is known at the
times t (sorted), with values u and derivatives ks
"""
@inline function interpolation(tval::Number, id::I, idxs, deriv::D, p,
                               continuity::Symbol = :left) where {I, D}
    t = id.t
    u = id.u
    typeof(id) <: HermiteInterpolation && (du = id.du)
    tdir = sign(t[end] - t[1])
    tdir * tval > tdir * t[end] &&
        error("Solution interpolation cannot extrapolate past the final timepoint. Either solve on a longer timespan or use the local extrapolation from the integrator interface.")
    tdir * tval < tdir * t[1] &&
        error("Solution interpolation cannot extrapolate before the first timepoint. Either start solving earlier or use the local extrapolation from the integrator interface.")
    @inbounds i = searchsortedfirst(t, tval, rev = tdir < 0) # It's in the interval t[i-1] to t[i]
    avoid_constant_ends = deriv != Val{0} #|| typeof(tval) <: ForwardDiff.Dual
    avoid_constant_ends && i == 1 && (i += 1)
    @inbounds if !avoid_constant_ends && t[i] == tval
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
        typeof(id) <: SensitivityInterpolation && error(SENSITIVITY_INTERP_MESSAGE)
        dt = t[i] - t[i - 1]
        Θ = (tval - t[i - 1]) / dt
        idxs_internal = idxs
        if typeof(id) <: HermiteInterpolation
            val = interpolant(Θ, id, dt, u[i - 1], u[i], du[i - 1], du[i], idxs_internal,
                              deriv)
        else
            val = interpolant(Θ, id, dt, u[i - 1], u[i], idxs_internal, deriv)
        end
    end
    val
end

"""
$(SIGNATURES)

Get the value at tval where the solution is known at the
times t (sorted), with values u and derivatives ks
"""
@inline function interpolation!(out, tval::Number, id::I, idxs, deriv::D, p,
                                continuity::Symbol = :left) where {I, D}
    t = id.t
    u = id.u
    typeof(id) <: HermiteInterpolation && (du = id.du)
    tdir = sign(t[end] - t[1])
    tdir * tval > tdir * t[end] &&
        error("Solution interpolation cannot extrapolate past the final timepoint. Either solve on a longer timespan or use the local extrapolation from the integrator interface.")
    tdir * tval < tdir * t[1] &&
        error("Solution interpolation cannot extrapolate before the first timepoint. Either start solving earlier or use the local extrapolation from the integrator interface.")
    @inbounds i = searchsortedfirst(t, tval, rev = tdir < 0) # It's in the interval t[i-1] to t[i]
    avoid_constant_ends = deriv != Val{0} #|| typeof(tval) <: ForwardDiff.Dual
    avoid_constant_ends && i == 1 && (i += 1)
    @inbounds if !avoid_constant_ends && t[i] == tval
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
        typeof(id) <: SensitivityInterpolation && error(SENSITIVITY_INTERP_MESSAGE)
        dt = t[i] - t[i - 1]
        Θ = (tval - t[i - 1]) / dt
        idxs_internal = idxs
        if typeof(id) <: HermiteInterpolation
            interpolant!(out, Θ, id, dt, u[i - 1], u[i], du[i - 1], du[i], idxs_internal,
                         deriv)
        else
            interpolant!(out, Θ, id, dt, u[i - 1], u[i], idxs_internal, deriv)
        end
    end
end

@inline function interpolant(Θ, id::AbstractDiffEqInterpolation, dt, y₀, y₁, dy₀, dy₁, idxs,
                             ::Type{Val{D}}) where {D}
    error("$(string(typeof(id))) for $(D)th order not implemented")
end
##################### Hermite Interpolants

"""
Hairer Norsett Wanner Solving Ordinary Differential Equations I - Nonstiff Problems Page 190

Hermite Interpolation
"""
@inline function interpolant(Θ, id::HermiteInterpolation, dt, y₀, y₁, dy₀, dy₁, idxs,
                             T::Type{Val{0}})
    if idxs === nothing
        out = @. (1 - Θ) * y₀ + Θ * y₁ +
                 Θ * (Θ - 1) * ((1 - 2Θ) * (y₁ - y₀) + (Θ - 1) * dt * dy₀ + Θ * dt * dy₁)
    elseif idxs isa Number
        out = (1 - Θ) * y₀[idxs] + Θ * y₁[idxs] +
              Θ * (Θ - 1) *
              ((1 - 2Θ) * (y₁[idxs] - y₀[idxs]) +
               (Θ - 1) * dt * dy₀[idxs] + Θ * dt * dy₁[idxs])
    else
        out = similar(y₀, axes(idxs))
        @views @. out = (1 - Θ) * y₀[idxs] + Θ * y₁[idxs] +
                        Θ * (Θ - 1) *
                        ((1 - 2Θ) * (y₁[idxs] - y₀[idxs]) +
                         (Θ - 1) * dt * dy₀[idxs] + Θ * dt * dy₁[idxs])
    end
    out
end

"""
Hermite Interpolation
"""
@inline function interpolant(Θ, id::HermiteInterpolation, dt, y₀, y₁, dy₀, dy₁, idxs,
                             T::Type{Val{1}})
    if idxs === nothing
        out = @. dy₀ +
                 Θ * (-4 * dt * dy₀ - 2 * dt * dy₁ - 6 * y₀ +
                  Θ * (3 * dt * dy₀ + 3 * dt * dy₁ + 6 * y₀ - 6 * y₁) + 6 * y₁) / dt
    elseif idxs isa Number
        out = dy₀[idxs] +
              Θ * (-4 * dt * dy₀[idxs] -
               2 * dt * dy₁[idxs] - 6 * y₀[idxs] +
               Θ * (3 * dt * dy₀[idxs] + 3 * dt * dy₁[idxs] +
                    6 * y₀[idxs] - 6 * y₁[idxs]) + 6 * y₁[idxs]) / dt
    else
        out = similar(y₀, axes(idxs))
        @views @. out = dy₀[idxs] +
                        Θ * (-4 * dt * dy₀[idxs] -
                         2 * dt * dy₁[idxs] - 6 * y₀[idxs] +
                         Θ * (3 * dt * dy₀[idxs] + 3 * dt * dy₁[idxs] +
                          6 * y₀[idxs] - 6 * y₁[idxs]) + 6 * y₁[idxs]) / dt
    end
    out
end

"""
Hermite Interpolation
"""
@inline function interpolant(Θ, id::HermiteInterpolation, dt, y₀, y₁, dy₀, dy₁, idxs,
                             T::Type{Val{2}})
    if idxs === nothing
        out = @. (-4 * dt * dy₀ - 2 * dt * dy₁ - 6 * y₀ +
                  Θ * (6 * dt * dy₀ + 6 * dt * dy₁ + 12 * y₀ - 12 * y₁) + 6 * y₁) /
                 (dt * dt)
    elseif idxs isa Number
        out = (-4 * dt * dy₀[idxs] - 2 * dt * dy₁[idxs] - 6 * y₀[idxs] +
               Θ * (6 * dt * dy₀[idxs] + 6 * dt * dy₁[idxs] + 12 * y₀[idxs] -
                12 * y₁[idxs]) + 6 * y₁[idxs]) / (dt * dt)
    else
        out = similar(y₀, axes(idxs))
        @views @. out = (-4 * dt * dy₀[idxs] - 2 * dt * dy₁[idxs] - 6 * y₀[idxs] +
                         Θ * (6 * dt * dy₀[idxs] + 6 * dt * dy₁[idxs] + 12 * y₀[idxs] -
                          12 * y₁[idxs]) + 6 * y₁[idxs]) / (dt * dt)
    end
    out
end

"""
Hermite Interpolation
"""
@inline function interpolant(Θ, id::HermiteInterpolation, dt, y₀, y₁, dy₀, dy₁, idxs,
                             T::Type{Val{3}})
    if idxs === nothing
        out = @. (6 * dt * dy₀ + 6 * dt * dy₁ + 12 * y₀ - 12 * y₁) / (dt * dt * dt)
    elseif idxs isa Number
        out = (6 * dt * dy₀[idxs] + 6 * dt * dy₁[idxs] +
               12 * y₀[idxs] - 12 * y₁[idxs]) / (dt * dt * dt)
    else
        out = similar(y₀, axes(idxs))
        @views @. out = (6 * dt * dy₀[idxs] + 6 * dt * dy₁[idxs] +
                         12 * y₀[idxs] - 12 * y₁[idxs]) / (dt * dt * dt)
    end
    out
end

"""
Hairer Norsett Wanner Solving Ordinary Differential Euations I - Nonstiff Problems Page 190

Hermite Interpolation
"""
@inline function interpolant!(out, Θ, id::HermiteInterpolation, dt, y₀, y₁, dy₀, dy₁, idxs,
                              T::Type{Val{0}})
    if out === nothing
        return (1 - Θ) * y₀[idxs] + Θ * y₁[idxs] +
               Θ * (Θ - 1) *
               ((1 - 2Θ) * (y₁[idxs] - y₀[idxs]) + (Θ - 1) * dt * dy₀[idxs] +
                Θ * dt * dy₁[idxs])
    elseif idxs === nothing
        @. out = (1 - Θ) * y₀ + Θ * y₁ +
                 Θ * (Θ - 1) * ((1 - 2Θ) * (y₁ - y₀) + (Θ - 1) * dt * dy₀ + Θ * dt * dy₁)
    else
        @views @. out = (1 - Θ) * y₀[idxs] + Θ * y₁[idxs] +
                        Θ * (Θ - 1) *
                        ((1 - 2Θ) * (y₁[idxs] - y₀[idxs]) + (Θ - 1) * dt * dy₀[idxs] +
                         Θ * dt * dy₁[idxs])
    end
end

"""
Hermite Interpolation
"""
@inline function interpolant!(out, Θ, id::HermiteInterpolation, dt, y₀, y₁, dy₀, dy₁, idxs,
                              T::Type{Val{1}})
    if out === nothing
        return dy₀[idxs] +
               Θ * (-4 * dt * dy₀[idxs] - 2 * dt * dy₁[idxs] - 6 * y₀[idxs] +
                Θ *
                (3 * dt * dy₀[idxs] + 3 * dt * dy₁[idxs] + 6 * y₀[idxs] - 6 * y₁[idxs]) +
                6 * y₁[idxs]) / dt
    elseif idxs === nothing
        @. out = dy₀ +
                 Θ * (-4 * dt * dy₀ - 2 * dt * dy₁ - 6 * y₀ +
                  Θ * (3 * dt * dy₀ + 3 * dt * dy₁ + 6 * y₀ - 6 * y₁) + 6 * y₁) / dt
    else
        @views @. out = dy₀[idxs] +
                        Θ * (-4 * dt * dy₀[idxs] - 2 * dt * dy₁[idxs] - 6 * y₀[idxs] +
                         Θ * (3 * dt * dy₀[idxs] + 3 * dt * dy₁[idxs] + 6 * y₀[idxs] -
                          6 * y₁[idxs]) + 6 * y₁[idxs]) / dt
    end
end

"""
Hermite Interpolation
"""
@inline function interpolant!(out, Θ, id::HermiteInterpolation, dt, y₀, y₁, dy₀, dy₁, idxs,
                              T::Type{Val{2}})
    if out === nothing
        return (-4 * dt * dy₀[idxs] - 2 * dt * dy₁[idxs] - 6 * y₀[idxs] +
                Θ *
                (6 * dt * dy₀[idxs] + 6 * dt * dy₁[idxs] + 12 * y₀[idxs] - 12 * y₁[idxs]) +
                6 * y₁[idxs]) / (dt * dt)
    elseif idxs === nothing
        @. out = (-4 * dt * dy₀ - 2 * dt * dy₁ - 6 * y₀ +
                  Θ * (6 * dt * dy₀ + 6 * dt * dy₁ + 12 * y₀ - 12 * y₁) + 6 * y₁) /
                 (dt * dt)
    else
        @views @. out = (-4 * dt * dy₀[idxs] - 2 * dt * dy₁[idxs] - 6 * y₀[idxs] +
                         Θ * (6 * dt * dy₀[idxs] + 6 * dt * dy₁[idxs] + 12 * y₀[idxs] -
                          12 * y₁[idxs]) + 6 * y₁[idxs]) / (dt * dt)
    end
end

"""
Hermite Interpolation
"""
@inline function interpolant!(out, Θ, id::HermiteInterpolation, dt, y₀, y₁, dy₀, dy₁, idxs,
                              T::Type{Val{3}})
    if out === nothing
        return (6 * dt * dy₀[idxs] + 6 * dt * dy₁[idxs] + 12 * y₀[idxs] - 12 * y₁[idxs]) /
               (dt * dt * dt)
    elseif idxs === nothing
        @. out = (6 * dt * dy₀ + 6 * dt * dy₁ + 12 * y₀ - 12 * y₁) / (dt * dt * dt)
    else
        @views @. out = (6 * dt * dy₀[idxs] + 6 * dt * dy₁[idxs] + 12 * y₀[idxs] -
                         12 * y₁[idxs]) / (dt * dt * dt)
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
    out
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
    out
end

"""
Linear Interpolation
"""
@inline function interpolant!(out, Θ, id::LinearInterpolation, dt, y₀, y₁, idxs,
                              T::Type{Val{0}})
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
@inline function interpolant!(out, Θ, id::LinearInterpolation, dt, y₀, y₁, idxs,
                              T::Type{Val{1}})
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
@inline function interpolant(Θ, id::ConstantInterpolation, dt, y₀, y₁, idxs,
                             T::Type{Val{0}})
    if idxs === nothing
        out = @. y₀
    elseif idxs isa Number
        out = y₀[idxs]
    else
        out = similar(y₀, axes(idxs))
        @views @. out = y₀[idxs]
    end
    out
end

@inline function interpolant(Θ, id::ConstantInterpolation, dt, y₀, y₁, idxs,
                             T::Type{Val{1}})
    if idxs === nothing
        out = zeros(eltype(y₀), length(y₀))
    elseif idxs isa Number
        out = zero(eltype(y₀))
    else
        out = similar(y₀, axes(idxs))
        @views @. out = 0
    end
    out
end

"""
Constant Interpolation
"""
@inline function interpolant!(out, Θ, id::ConstantInterpolation, dt, y₀, y₁, idxs,
                              T::Type{Val{0}})
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
@inline function interpolant!(out, Θ, id::ConstantInterpolation, dt, y₀, y₁, idxs,
                              T::Type{Val{1}})
    if out === nothing
        return zeros(eltype(y₀), length(idxs))
    else
        @. out = 0
    end
end
