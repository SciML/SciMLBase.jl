"""
$(TYPEDEF)

Base interface for internal one-variable views of SciML model functions.

Wrapped functions close over all but one argument of a model function so
automatic differentiation or finite-difference code can differentiate with
respect to time, state, or parameters. The `iip` type parameter preserves the
in-place convention of the wrapped function and is returned by [`isinplace`](@ref).
"""
abstract type AbstractWrappedFunction{iip} end
isinplace(f::AbstractWrappedFunction{iip}) where {iip} = iip

"""
$(TYPEDEF)

Expose time as the only free argument of an ODE-style model function.

`TimeGradientWrapper(f, uprev, p)` fixes the state and parameters of `f` and
returns a callable object over `t`. For out-of-place functions it calls
`f(uprev, p, t)`. For in-place functions, `wrapper(out, t)` calls
`f(out, uprev, p, t)`, while `wrapper(t)` allocates `similar(uprev)` and returns
the filled value. This form is used when AD code needs a one-argument function
for the time gradient.

# Fields

$(TYPEDFIELDS)
"""
mutable struct TimeGradientWrapper{iip, fType, uType, P} <: AbstractWrappedFunction{iip}
    f::fType
    uprev::uType
    p::P
end

function TimeGradientWrapper{iip}(f::F, uprev, p) where {F, iip}
    return TimeGradientWrapper{iip, F, typeof(uprev), typeof(p)}(f, uprev, p)
end
function TimeGradientWrapper(f::F, uprev, p) where {F}
    return TimeGradientWrapper{isinplace(f, 4)}(f, uprev, p)
end

function (ff::TimeGradientWrapper{true})(t)
    (du2 = similar(ff.uprev); ff.f(du2, ff.uprev, ff.p, t); du2)
end
(ff::TimeGradientWrapper{true})(du2, t) = ff.f(du2, ff.uprev, ff.p, t)

(ff::TimeGradientWrapper{false})(t) = ff.f(ff.uprev, ff.p, t)

"""
$(TYPEDEF)

Expose the state as the free argument of an ODE-style model function.

`UJacobianWrapper(f, t, p)` fixes time and parameters and returns a callable
object over `u`. For out-of-place functions it calls `f(u, p, t)`. For in-place
functions, `wrapper(out, u)` calls `f(out, u, p, t)`, while `wrapper(u)`
allocates `similar(u)` and returns the filled value.

The overloads `wrapper(u, p, t)` and `wrapper(out, u, p, t)` let callers reuse
the same wrapper while overriding the closed-over parameters and time.

# Fields

$(TYPEDFIELDS)
"""
mutable struct UJacobianWrapper{iip, fType, tType, P} <: AbstractWrappedFunction{iip}
    f::fType
    t::tType
    p::P
end

function UJacobianWrapper{iip}(f::F, t, p) where {F, iip}
    return UJacobianWrapper{iip, F, typeof(t), typeof(p)}(f, t, p)
end
UJacobianWrapper(f::F, t, p) where {F} = UJacobianWrapper{isinplace(f, 4)}(f, t, p)

(ff::UJacobianWrapper{true})(du1, uprev) = ff.f(du1, uprev, ff.p, ff.t)
function (ff::UJacobianWrapper{true})(uprev)
    (du1 = similar(uprev); ff.f(du1, uprev, ff.p, ff.t); du1)
end
(ff::UJacobianWrapper{true})(du1, uprev, p, t) = ff.f(du1, uprev, p, t)
function (ff::UJacobianWrapper{true})(uprev, p, t)
    (du1 = similar(uprev); ff.f(du1, uprev, p, t); du1)
end

(ff::UJacobianWrapper{false})(uprev) = ff.f(uprev, ff.p, ff.t)
(ff::UJacobianWrapper{false})(uprev, p, t) = ff.f(uprev, p, t)

"""
$(TYPEDEF)

Fix state and parameters while exposing time as the differentiated variable.

`TimeDerivativeWrapper(f, u, p)` is the fixed-state, fixed-parameter view used
by derivative code that expects a function of `t` alone. For out-of-place
functions it calls `f(u, p, t)`. For in-place functions, `wrapper(out, t)` calls
`f(out, u, p, t)`, while `wrapper(t)` allocates `similar(u)` and returns the
filled value.

# Fields

$(TYPEDFIELDS)
"""
mutable struct TimeDerivativeWrapper{iip, F, uType, P} <: AbstractWrappedFunction{iip}
    f::F
    u::uType
    p::P
end

function TimeDerivativeWrapper{iip}(f::F, u, p) where {F, iip}
    return TimeDerivativeWrapper{iip, F, typeof(u), typeof(p)}(f, u, p)
end
function TimeDerivativeWrapper(f::F, u, p) where {F}
    return TimeDerivativeWrapper{isinplace(f, 4)}(f, u, p)
end

(ff::TimeDerivativeWrapper{false})(t) = ff.f(ff.u, ff.p, t)
(ff::TimeDerivativeWrapper{true})(du1, t) = ff.f(du1, ff.u, ff.p, t)
(ff::TimeDerivativeWrapper{true})(t) = (du1 = similar(ff.u); ff.f(du1, ff.u, ff.p, t); du1)

"""
$(TYPEDEF)

Fix time and parameters while exposing the state as the differentiated variable.

`UDerivativeWrapper(f, t, p)` is the fixed-time, fixed-parameter view used by
derivative code that expects a function of `u` alone. For out-of-place functions
it calls `f(u, p, t)`. For in-place functions, `wrapper(out, u)` calls
`f(out, u, p, t)`, while `wrapper(u)` allocates `similar(u)` and returns the
filled value.

# Fields

$(TYPEDFIELDS)
"""
mutable struct UDerivativeWrapper{iip, F, tType, P} <: AbstractWrappedFunction{iip}
    f::F
    t::tType
    p::P
end

function UDerivativeWrapper{iip}(f::F, t, p) where {F, iip}
    return UDerivativeWrapper{iip, F, typeof(t), typeof(p)}(f, t, p)
end
UDerivativeWrapper(f::F, t, p) where {F} = UDerivativeWrapper{isinplace(f, 4)}(f, t, p)

(ff::UDerivativeWrapper{false})(u) = ff.f(u, ff.p, ff.t)
(ff::UDerivativeWrapper{true})(du1, u) = ff.f(du1, u, ff.p, ff.t)
(ff::UDerivativeWrapper{true})(u) = (du1 = similar(u); ff.f(du1, u, ff.p, ff.t); du1)

"""
    ParamJacobianWrapper(f, t, u)
    ParamJacobianWrapper{iip}(f, t, u)

Wrap an ODE-style function as a function of parameters alone.

# Arguments

  - `f`: An in-place `f(du, u, p, t)` or out-of-place `f(u, p, t)` model function.
  - `t`: The fixed independent-variable value.
  - `u`: The fixed state value.
  - `iip`: Whether `f` follows the in-place convention. The unparameterized constructor
    infers it from `f`.

# Fields

  - `f`: The wrapped model function.
  - `t`: The fixed independent-variable value.
  - `u`: The fixed state value.

# Type Parameters

  - `iip`: Whether the wrapped function is in-place.
  - `fType`: Type of the wrapped function.
  - `tType`: Type of the fixed independent-variable value.
  - `uType`: Type of the fixed state value.

# Usage

```julia
pf = ParamJacobianWrapper((u, p, t) -> p .* u, 0.0, [2.0])
pf([3.0]) # [6.0]
```

# Developer Interface

Sensitivity and solver packages use this wrapper when differentiating a model with
respect to `p`. Call `pf(p)` to allocate an output or `pf(out, p)` to fill a provided
output buffer. The fixed `u` and `t` values must be valid inputs for `f`; downstream
code must use the callable interface rather than depending on the mutable field layout.
"""
mutable struct ParamJacobianWrapper{iip, fType, tType, uType} <: AbstractWrappedFunction{iip}
    f::fType
    t::tType
    u::uType
end

function ParamJacobianWrapper{iip}(f::F, t, u) where {F, iip}
    return ParamJacobianWrapper{iip, F, typeof(t), typeof(u)}(f, t, u)
end
ParamJacobianWrapper(f::F, t, u) where {F} = ParamJacobianWrapper{isinplace(f, 4)}(f, t, u)

(ff::ParamJacobianWrapper{true})(du1, p) = ff.f(du1, ff.u, p, ff.t)
function (ff::ParamJacobianWrapper{true})(p)
    du1 = similar(p, size(ff.u))
    ff.f(du1, ff.u, p, ff.t)
    return du1
end
(ff::ParamJacobianWrapper{false})(p) = ff.f(ff.u, p, ff.t)
function (ff::ParamJacobianWrapper{false})(du1, p)
    return du1 .= ff.f(ff.u, p, ff.t)
end

"""
    JacobianWrapper(f, p)
    JacobianWrapper{iip}(f, p)

Fix the parameter argument of a residual function and expose the state as its free
argument.

# Arguments

- `f`: An in-place `f(residual, u, p)` or out-of-place `f(u, p)` function.
- `p`: The fixed parameter value.
- `iip`: Whether `f` follows the in-place convention. The unparameterized constructor
  infers this from `f`.

# Fields

- `f`: The wrapped residual function.
- `p`: The fixed parameter value.

# Type Parameters

- `iip`: Whether the wrapped function is in-place.
- `fType`: Type of the wrapped function.
- `pType`: Type of the fixed parameter value.

# Returns

Calling an out-of-place wrapper as `wrapper(u)` returns `f(u, p)`. Calling either
convention as `wrapper(residual, u)` fills `residual`; the in-place form returns the
return value of `f`, while the out-of-place form returns the broadcast assignment.

# Example

```julia
wrapper = JacobianWrapper((u, p) -> u .- p, [1.0, 2.0])
wrapper([3.0, 5.0])
```

# Developer Interface

Nonlinear solver and differentiation packages may construct this wrapper when they need
a one-argument residual with fixed parameters. Use its callable interface rather than
depending on the mutable field layout.
"""
mutable struct JacobianWrapper{iip, fType, pType} <: AbstractWrappedFunction{iip}
    f::fType
    p::pType
end

JacobianWrapper{iip}(f::F, p) where {F, iip} = JacobianWrapper{iip, F, typeof(p)}(f, p)
JacobianWrapper(f::F, p) where {F} = JacobianWrapper{isinplace(f, 3)}(f, p)

(uf::JacobianWrapper{false})(u) = uf.f(u, uf.p)
(uf::JacobianWrapper{false})(res, u) = (vec(res) .= vec(uf.f(u, uf.p)))
(uf::JacobianWrapper{true})(res, u) = uf.f(res, u, uf.p)
