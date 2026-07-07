"""
    EigenvalueTarget

Enum selecting which part of the spectrum is returned when only a subset of the
eigenpairs is requested (via `num_eigenpairs`) in an [`EigenvalueProblem`](@ref).
"""
EnumX.@enumx EigenvalueTarget begin
    "Eigenvalues of largest magnitude, `abs(λ)` largest."
    LargestMagnitude
    "Eigenvalues of smallest magnitude, `abs(λ)` smallest."
    SmallestMagnitude
    "Eigenvalues with the largest (most positive) real part."
    LargestRealPart
    "Eigenvalues with the smallest (most negative) real part."
    SmallestRealPart
    "Eigenvalues with the largest (most positive) imaginary part."
    LargestImaginaryPart
    "Eigenvalues with the smallest (most negative) imaginary part."
    SmallestImaginaryPart
end

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

### Type Promotion Rules

  - The eigenvector type follows `u0` when it is supplied: `typeof(v) == typeof(u0)`.
    Otherwise, eigenvectors are returned as the dense vector type corresponding to a row
    of `A` (e.g. sparse `A` still returns dense eigenvectors).
  - The eigenvalue type follows `eltype(A)`: `typeof(lambda) == eltype(A)` whenever the
    eigenvalues are real (e.g. `A` symmetric or Hermitian). For a general, non-symmetric
    real `A`, eigenvalues may be complex conjugate pairs, in which case
    `typeof(lambda) == Complex{eltype(A)}`.

## Problem Type

### Constructors

```julia
EigenvalueProblem(A, B = nothing, p = NullParameters();
    num_eigenpairs = nothing, eigentarget = EigenvalueTarget.LargestMagnitude,
    shift = nothing, u0 = nothing, kwargs...)
```

### Keyword Arguments

  - `num_eigenpairs`: the number of eigenpairs (eigenvalues together with their
    eigenvectors) to compute. `nothing` (the default) requests every eigenpair for the
    dense solver, or a solver-chosen default for the iterative backends.
  - `eigentarget`: which part of the spectrum to return, as an [`EigenvalueTarget`](@ref).
    Defaults to the eigenvalues of largest magnitude.
  - `shift`: if supplied, return the eigenvalues nearest this shift (shift-and-invert).
  - `u0`: optional initial guess for the iterative backends.

Any extra keyword arguments are passed on to the solver.

### Fields

  - `A`: the operator whose eigenvalues are sought.
  - `B`: the second operator for a generalized problem, or `nothing` for a standard one.
  - `num_eigenpairs`: the requested number of eigenpairs.
  - `eigentarget`: the [`EigenvalueTarget`](@ref) selecting the part of the spectrum.
  - `shift`: the shift for shift-and-invert, or `nothing`.
  - `u0`: the initial guess used by iterative solvers.
  - `p`: the parameters for the problem. Defaults to `NullParameters`.
  - `kwargs`: the keyword arguments passed on to the solvers.
"""
struct EigenvalueProblem{
        AType, BType, NevType, TargetType, ShiftType, U0Type, PType, KType,
    } <: AbstractEigenvalueProblem
    A::AType
    B::BType
    num_eigenpairs::NevType
    eigentarget::TargetType
    shift::ShiftType
    u0::U0Type
    p::PType
    kwargs::KType
end

function EigenvalueProblem(
        A, B = nothing, p = NullParameters();
        num_eigenpairs = nothing, eigentarget::EigenvalueTarget.T = EigenvalueTarget.LargestMagnitude,
        shift = nothing, u0 = nothing, kwargs...
    )
    warn_paramtype(p)
    return EigenvalueProblem{
        typeof(A), typeof(B), typeof(num_eigenpairs), typeof(eigentarget), typeof(shift),
        typeof(u0), typeof(p), typeof(kwargs),
    }(A, B, num_eigenpairs, eigentarget, shift, u0, p, kwargs)
end
