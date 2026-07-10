# [SciML Container (Array) and Number Interfaces](@id arrayandnumber)

SciML problems separate the mathematical model from the numerical algorithm,
so compatibility is a property of the complete problem-algorithm pair. A type
that is sufficient for an explicit out-of-place method may be insufficient for
an in-place, adaptive, implicit, sparse, GPU, or directly differentiated solve.
This page defines the shared contracts and identifies the features that add
further requirements.

The contracts below are written as closure laws on the scalar type `T`, the
container type `C`, the time type `T2`, and the matrix/operator type `M`. A law
of the form `x::T + y::T = z::T` means the operation must be defined *and* must
return the same type, not merely a value: type preservation is part of the
contract, because solver caches, tableaus, and interpolants are typed off these
results. Where a feature is optional, that is stated explicitly; everything else
is required for the corresponding solve path.

## Algorithm Capabilities

Before relying on a nonstandard scalar or container, check the selected solver
package and algorithm. SciMLBase defines traits that solver packages extend:

  - [`allows_arbitrary_number_types(alg)`](@ref SciMLBase.allows_arbitrary_number_types)
    declares support for scalar types beyond standard floating-point and complex
    floating-point types.
  - [`allowscomplex(alg)`](@ref SciMLBase.allowscomplex) declares support for
    complex-valued states. Complex-valued time spans are not supported.
  - [`isautodifferentiable(alg)`](@ref SciMLBase.isautodifferentiable) declares
    whether automatic differentiation may pass directly through the solver.

These traits describe algorithm implementations, not just the user model. A
solver can accept a model that evaluates on dual numbers without supporting
direct differentiation through all of its caches, callbacks, and linear solves.
Likewise, an algorithm that supports arbitrary scalar types may still require a
specific array layout or linear solver for a particular configuration.

## Wrapped Solvers

Solvers that call C, Fortran, or another external runtime inherit that library's
ABI and storage constraints. These commonly include a fixed real scalar type,
contiguous host storage, or a specific vector representation. The precise
restriction belongs to the wrapper: it is not a universal rule that every
wrapped solver accepts only `Vector{Float64}`, and callers should consult the
selected package's documented problem and algorithm support.

A wrapper may copy or convert compatible inputs to its native representation.
Such conversion does not imply support for units, dual numbers, arbitrary
precision, GPU arrays, custom axes, or preservation of the input container type.
Use an algorithm that advertises the required capabilities when those properties
must survive the solve.

## SciML Number Types

The number type `T` is used for the dependent variables (i.e. `u0`) and, through
the time type `T2`, for the independent variable (`t`/`tspan`). `T` and `T2` may
differ and carry different restrictions. If a problem defines a value such as
`u0` with a bare number type, the out-of-place problem form must be used.

The following laws hold for a number type `T` in general:

  - `zero(x::T)::T` and `oneunit(x::T)::T`
  - `(one(x::T) * oneunit(x::T))::T`
  - `x::T + y::T = z::T`
  - `x::T - y::T = z::T`
  - `x::T * y::T = z::T`
  - `t::T2 * x::T + y::T = z::T` for `T2` a time type (this includes the
    `muladd` equivalent form)
  - Promotion is closed on `T`: when a solver coefficient of type `S` (commonly
    `Float64` or `Rational`) meets a value of type `T`, `promote_type(T, S)`
    must be a type the solve can continue in — for a type intended to be
    preserved, that promotion is `T` itself.

Additionally, the following laws apply to subsets of uses.

### Adaptive Number Types

Adaptive error control requires, on top of the general laws:

  - `x::T / y::T = z::T`
  - `x::T ^ y::T = z::T`
  - Default choices of norms may assume `sqrt(x::T)::T` exists. If
    `internalnorm` is overridden this may not be required (for example, changing
    the norm to the inf-norm).
  - The error estimate and `internalnorm` produce a finite, ordered, effectively
    dimensionless value that supports `<` against the tolerances.

Overriding `internalnorm` can make a custom scalar or container usable when the
default norm is inappropriate, but it does not remove the other arithmetic
requirements of the algorithm, and tolerances must remain dimensionally
compatible with the scaled error calculation.

### Time Types (Independent Variables)

The time type `T2` is a real, ordered number type. It must provide:

  - `t1::T2 - t2::T2 = dt::T2` and `t::T2 + dt::T2 = t3::T2`
  - ordering (`<`), `sign`, and `iszero` for direction and termination tests
  - `c * dt::T2 = dt2::T2` for a dimensionless scalar `c`

If a solver is time adaptive, the time type must be a floating point number.
`Rational` is only allowed for non-adaptive solves. Exact types such as
`Rational` are preserved by compatible fixed-step methods but are not a general
promise for adaptive methods. Pure-Julia algorithms may also support `BigFloat`,
dual-valued time, or Unitful time when all selected features remain generic; the
algorithm traits and package tests are authoritative for a specific combination.

## SciML Container (Array) Types

Container types hold number types and are used to define objects such as the
state vector `u0`. A container type `C` with element type `E` used with SciML
solvers must satisfy:

  - Broadcast is defined
    [according to the Julia broadcast interface](https://docs.julialang.org/en/v1/manual/interfaces/#man-interfaces-broadcasting),
    and the container implements Julia's
    [array interface](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array).
  - The container correctly defines the public
    [ArrayInterface.jl](https://docs.sciml.ai/ArrayInterface/stable/) trait
    overloads. In particular `ArrayInterface.ismutable(C)` must agree with
    whether solver caches may write into the value.
  - `eltype(x::C)::E` is a compatible `Number` type.
  - `size`, `axes`, `length`, and `eachindex` report the state geometry.
  - `ArrayInterface.zeromatrix(x::C)::M` defines a compatible matrix type (see
    below).
  - `x::C .+ y::C = z::C` (broadcast-`similar` is type-preserving).
  - Scalar indexing `x[i]` is required only if
    `ArrayInterface.fast_scalar_indexing(x::C) == true`; when true it must be
    defined and iterate over all variables. Declaring the trait `false` does not
    by itself make scalar-indexing solver code GPU-compatible.

!!! note

    "`eltype(x::C)::E` is a compatible `Number` type" excludes `Array{Array{T}}`
    types of types. Recursive vectors can be conformed to the interface with zero
    overhead using tools from RecursiveArrayTools.jl such as `VectorOfArray(x)`
    or `ArrayPartition`, or a structured flat container such as `ComponentArray`.
    Since this greatly simplifies the interfaces and the ability to check for
    correctness, doing this wrapping is highly recommended and there are no plans
    to relax this requirement. Boundary value problems have additional documented
    forms for arrays of state guesses.

Additionally, the following laws apply to subsets of uses.

### SciML Mutable Array Types

In-place models and mutable solver caches require writable, type-preserving
containers. On top of the general container laws:

  - `similar(x::C)::C`, along with `similar(x::C, E)` and `similar(x::C, dims)`
    for requested element types or geometry
  - `zero(x::C)::C`
  - `copy(x::C)::C`, `copyto!(dest::C, src::C)`, and `fill!(x::C, v::E)`
  - `z::C .= x::C .+ y::C` is defined
  - `z::C .= x::C .* y::C` is defined
  - `z::C .= t::T2 .* x::C` where `T2` is the time type
  - `getindex` and `setindex!` when the algorithm uses scalar indexing
  - (Optional) `Base.resize!(x, i)` is required for `resize!(integrator, i)` to
    be supported

Immutable containers such as `SVector` use the out-of-place model and cache
paths instead. GPU arrays additionally require an algorithm, differentiation
backend, callback set, and linear solver that all avoid unsupported host scalar
indexing.

### SciML Matrix (Operator) Type

The matrix type `M` may not match the type of the initial container `u0`. An
example is `ComponentMatrix` as the matrix structure corresponding to a
`ComponentArray`; `ArrayInterface.zeromatrix(u)` selects the compatible default
with the correct dimensions and storage location. The following actions are
assumed to hold on the resulting matrix type:

  - `solve(LinearProblem(A::M, b::C), linsolve)` must be defined for a solver to
    work on a given SciML matrix type `M`, with `b` using the state-compatible
    right-hand-side type.
  - `mul!(y, A::M, x)` (matrix-vector product) and type-preserving `similar` and
    `copy` on `M`.
  - If the matrix is an operator, i.e. a lazy construct, it should conform to the
    public [SciMLOperators.jl](https://docs.sciml.ai/SciMLOperators/stable/)
    interface rather than imitating a dense array.
  - If not a SciMLOperator, `diagind(W::M)` should be defined and
    `@view(A[idxs]) = @view(A[idxs]) + λ::E` must be supported so that the shift
    used to form a Newton matrix can be applied.

Supplying `jac_prototype`, `linsolve`, or a mass matrix can therefore add
requirements beyond those of the state container alone.

## Validating a New Type

There is no supported keyword that bypasses all type and interface validation.
Test the exact problem-algorithm pair and every feature that will be used:
`init`, `solve`, interpolation, callbacks, adaptivity, Jacobian construction,
linear solves, device execution, and differentiation can exercise different
parts of the contract. A successful constructor call alone does not establish
solver compatibility.

Type authors should keep these tests in the package that owns the container or
number type and cover both result correctness and preservation of scalar,
container, axes, and device properties where those are part of the type's public
contract.
