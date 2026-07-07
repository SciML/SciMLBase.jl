"""
    EigenvalueTarget

Enum selecting which part of the spectrum is returned when only a subset of the
eigenpairs is requested (via `nev`) in an [`EigenvalueProblem`](@ref). The `which`
keyword accepts either an `EigenvalueTarget` value or, for convenience, the
corresponding ARPACK-style `Symbol` noted for each variant below.
"""
EnumX.@enumx EigenvalueTarget begin
    "Eigenvalues of largest magnitude, `abs(λ)` largest (symbol `:LM`)."
    LargestMagnitude
    "Eigenvalues of smallest magnitude, `abs(λ)` smallest (symbol `:SM`)."
    SmallestMagnitude
    "Eigenvalues with the largest (most positive) real part (symbol `:LR`)."
    LargestRealPart
    "Eigenvalues with the smallest (most negative) real part (symbol `:SR`)."
    SmallestRealPart
    "Eigenvalues with the largest (most positive) imaginary part (symbol `:LI`)."
    LargestImaginaryPart
    "Eigenvalues with the smallest (most negative) imaginary part (symbol `:SI`)."
    SmallestImaginaryPart
end

# Normalize a user-supplied `which` (an `EigenvalueTarget` or an ARPACK-style
# `Symbol`) to an `EigenvalueTarget`, throwing on anything else.
_eigenvalue_target(w::EigenvalueTarget.T) = w
function _eigenvalue_target(w::Symbol)
    return w === :LM ? EigenvalueTarget.LargestMagnitude :
        w === :SM ? EigenvalueTarget.SmallestMagnitude :
        w === :LR ? EigenvalueTarget.LargestRealPart :
        w === :SR ? EigenvalueTarget.SmallestRealPart :
        w === :LI ? EigenvalueTarget.LargestImaginaryPart :
        w === :SI ? EigenvalueTarget.SmallestImaginaryPart :
        throw(ArgumentError("unsupported eigenvalue selector `which = $(repr(w))`; expected an `EigenvalueTarget` or one of :LM, :SM, :LR, :SR, :LI, :SI"))
end
_eigenvalue_target(w) = throw(ArgumentError("`which` must be an `EigenvalueTarget` or a Symbol (:LM, :SM, :LR, :SR, :LI, :SI), got $(repr(w))"))

@doc doc"""

Defines a standard or generalized eigenvalue problem.

## Mathematical Specification of an Eigenvalue Problem

The standard problem finds pairs ``(\lambda, v)`` satisfying

```math
A v = \lambda v
```

If a second operator `B` is supplied, the generalized problem is solved instead:

```math
A v = \lambda B v
```

## Problem Type

### Constructors

```julia
EigenvalueProblem(A, B = nothing, p = NullParameters();
    nev = nothing, which = EigenvalueTarget.LargestMagnitude,
    sigma = nothing, u0 = nothing, kwargs...)
```

### Keyword Arguments

  - `nev`: the number of eigenpairs (eigenvalues together with their eigenvectors) to
    compute. `nothing` (the default) requests every eigenpair for the dense solver, or
    a solver-chosen default for the iterative backends.
  - `which`: which part of the spectrum to return, as an [`EigenvalueTarget`](@ref). An
    ARPACK-style `Symbol` (`:LM`, `:SM`, `:LR`, `:SR`, `:LI`, `:SI`) is also accepted and
    converted to the corresponding `EigenvalueTarget`. Defaults to the eigenvalues of
    largest magnitude.
  - `sigma`: if supplied, return the eigenvalues nearest this shift (shift-and-invert).
  - `u0`: optional initial guess for the iterative backends.

Any extra keyword arguments are passed on to the solver.

### Fields

  - `A`: the operator whose eigenvalues are sought.
  - `B`: the second operator for a generalized problem, or `nothing` for a standard one.
  - `nev`: the requested number of eigenpairs.
  - `which`: the [`EigenvalueTarget`](@ref) selecting the part of the spectrum.
  - `sigma`: the shift for shift-and-invert, or `nothing`.
  - `u0`: the initial guess used by iterative solvers.
  - `p`: the parameters for the problem. Defaults to `NullParameters`.
  - `kwargs`: the keyword arguments passed on to the solvers.
"""
struct EigenvalueProblem{
        AType, BType, NevType, WhichType, SigmaType, U0Type, PType, KType,
    } <: AbstractEigenvalueProblem
    A::AType
    B::BType
    nev::NevType
    which::WhichType
    sigma::SigmaType
    u0::U0Type
    p::PType
    kwargs::KType
end

function EigenvalueProblem(
        A, B = nothing, p = NullParameters();
        nev = nothing, which = EigenvalueTarget.LargestMagnitude,
        sigma = nothing, u0 = nothing, kwargs...
    )
    warn_paramtype(p)
    target = _eigenvalue_target(which)
    return EigenvalueProblem{
        typeof(A), typeof(B), typeof(nev), typeof(target), typeof(sigma),
        typeof(u0), typeof(p), typeof(kwargs),
    }(A, B, nev, target, sigma, u0, p, kwargs)
end
