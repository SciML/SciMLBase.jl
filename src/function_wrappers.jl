"""
    AbstractWrappedFunction{iip}

Abstract base type for function wrappers used in automatic differentiation and sensitivity analysis.
These wrappers provide specialized interfaces for computing derivatives with respect to different variables.

# Type Parameter
- `iip`: Boolean indicating if the function is in-place (`true`) or out-of-place (`false`)
"""
abstract type AbstractWrappedFunction{iip} end
isinplace(f::AbstractWrappedFunction{iip}) where {iip} = iip

"""
    TimeGradientWrapper{iip, fType, uType, P} <: AbstractWrappedFunction{iip}

Wraps functions to compute gradients with respect to time. This wrapper is particularly useful for 
sensitivity analysis and optimization problems where the time dependence of the solution is critical.

# Fields
- `f`: The function to wrap
- `uprev`: Previous state value
- `p`: Parameters

# Type Parameters
- `iip`: Boolean indicating if the function is in-place (`true`) or out-of-place (`false`)
- `fType`: Type of the wrapped function
- `uType`: Type of the state variables
- `P`: Type of the parameters

This wrapper enables automatic differentiation with respect to time by providing a consistent
interface for computing `∂f/∂t` across different AD systems.
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
    UJacobianWrapper{iip, fType, tType, P} <: AbstractWrappedFunction{iip}

Wraps functions to compute Jacobians with respect to state variables `u`. This is one of the most 
commonly used wrappers in the SciML ecosystem for computing the derivative of the right-hand side 
function with respect to the state variables.

# Fields
- `f`: The function to wrap
- `t`: Time value
- `p`: Parameters

# Type Parameters
- `iip`: Boolean indicating if the function is in-place (`true`) or out-of-place (`false`)
- `fType`: Type of the wrapped function
- `tType`: Type of the time variable
- `P`: Type of the parameters

This wrapper enables efficient computation of `∂f/∂u` for Jacobian calculations in numerical solvers
and automatic differentiation systems.
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
    TimeDerivativeWrapper{iip, F, uType, P} <: AbstractWrappedFunction{iip}

Wraps functions to compute derivatives with respect to time. This wrapper is used when you need to 
compute `∂f/∂t` for sensitivity analysis or when the function has explicit time dependence.

# Fields
- `f`: The function to wrap
- `u`: State variables
- `p`: Parameters

# Type Parameters
- `iip`: Boolean indicating if the function is in-place (`true`) or out-of-place (`false`)
- `F`: Type of the wrapped function
- `uType`: Type of the state variables
- `P`: Type of the parameters

This wrapper provides a consistent interface for time derivative computations across different
automatic differentiation backends.
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
    UDerivativeWrapper{iip, F, tType, P} <: AbstractWrappedFunction{iip}

Wraps functions to compute derivatives with respect to state variables. This wrapper is used for 
computing `∂f/∂u` and is fundamental for Jacobian computations in numerical solvers.

# Fields
- `f`: The function to wrap
- `t`: Time value
- `p`: Parameters

# Type Parameters
- `iip`: Boolean indicating if the function is in-place (`true`) or out-of-place (`false`)
- `F`: Type of the wrapped function
- `tType`: Type of the time variable
- `P`: Type of the parameters

This wrapper enables efficient state derivative computations for use in automatic differentiation
and numerical analysis algorithms.
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
    ParamJacobianWrapper{iip, fType, tType, uType} <: AbstractWrappedFunction{iip}

Wraps functions to compute Jacobians with respect to parameters `p`. This wrapper is essential for 
parameter estimation, inverse problems, and sensitivity analysis with respect to model parameters.

# Fields
- `f`: The function to wrap
- `t`: Time value
- `u`: State variables

# Type Parameters
- `iip`: Boolean indicating if the function is in-place (`true`) or out-of-place (`false`)
- `fType`: Type of the wrapped function
- `tType`: Type of the time variable
- `uType`: Type of the state variables

This wrapper enables efficient computation of `∂f/∂p` for parameter sensitivity analysis and
optimization algorithms.
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
    JacobianWrapper{iip, fType, pType} <: AbstractWrappedFunction{iip}

A general-purpose Jacobian wrapper that can be configured for different types of Jacobian computations. 
This wrapper provides a unified interface for various Jacobian calculations across the SciML ecosystem.

# Fields
- `f`: The function to wrap
- `p`: Parameters

# Type Parameters
- `iip`: Boolean indicating if the function is in-place (`true`) or out-of-place (`false`)
- `fType`: Type of the wrapped function
- `pType`: Type of the parameters

This wrapper provides a flexible interface for Jacobian computations that can adapt to different
automatic differentiation backends and numerical methods.
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
