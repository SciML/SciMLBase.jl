# [SciML Container (Array) and Number Interfaces](@id arrayandnumber)

SciML problems separate the mathematical model from the numerical algorithm,
so compatibility is a property of the complete problem-algorithm pair. A type
that is sufficient for an explicit out-of-place method may be insufficient for
an in-place, adaptive, implicit, sparse, GPU, or directly differentiated solve.
This page defines the shared contracts and identifies the features that add
further requirements.

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

## Scalar Number Types

State scalars and array element types must support the arithmetic performed by
the selected algorithm. The common baseline is:

  - additive and multiplicative identities through `zero`, `one`, and
    `oneunit` where dimensionful quantities require it;
  - addition, subtraction, multiplication, and the required division between
    states, rates, time steps, tableau coefficients, and parameters;
  - promotion or conversion that preserves the intended scalar type when solver
    coefficients and problem values interact; and
  - magnitude and comparison operations needed by the chosen norm, stopping
    conditions, and error checks.

The model must return rates with dimensions compatible with state divided by
time. It should not allocate hard-coded `Float64` work arrays when it is expected
to support `Float32`, `BigFloat`, unitful quantities, dual numbers, measurements,
or another scalar type. Use values and storage derived from the model arguments,
such as `zero(u)`, `similar(u)`, and `oneunit(t)`.

A scalar state is compatible with an out-of-place model. An in-place model needs
a writable state and derivative container; SciMLBase rejects scalar and immutable
`SArray` initial states for in-place differential equation problems.

### Adaptive Error Control

Adaptive algorithms impose additional operations. Their error estimate and
`internalnorm` must produce a finite, ordered, effectively dimensionless value
that can be compared with tolerances. Step-size controllers may apply division,
fractional powers or roots, minima and maxima, and conversion between controller
coefficients and the time-step type.

Overriding `internalnorm` can make a custom scalar or container usable when the
default norm is inappropriate, but it does not remove the other arithmetic
requirements of the algorithm. Tolerances must also be dimensionally compatible
with the scaled error calculation.

### Time Types

Time is real and ordered. A supported time type must provide subtraction between
times, addition of a time step, direction and zero tests, ordering, and scaling
by the dimensionless values used by the algorithm. Adaptive methods require the
extra controller operations described above.

Exact types such as `Rational` can be preserved by compatible fixed-step methods;
they are not a general promise for adaptive methods. Pure-Julia algorithms may
also support `BigFloat`, dual-valued time, or Unitful time when all selected
features remain generic. The algorithm traits and package tests are authoritative
for a specific combination.

## State Container Types

Array states should implement Julia's
[array interface](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array)
and [broadcasting interface](https://docs.julialang.org/en/v1/manual/interfaces/#man-interfaces-broadcasting).
At minimum, the solver must be able to inspect the state geometry and scalar
type through operations such as `size`, `axes`, `length`, `eltype`, and
`eachindex`, and out-of-place arithmetic must construct a result with compatible
geometry.

For an ordinary Julia `Array`, the differential equation solve path requires a
concrete recursive unitless scalar type and a `Number` element type. A plain
`Array{Array{T}}` is therefore not a general SciML state representation. Use a
container with defined recursive array semantics, such as `ArrayPartition` or
`VectorOfArray` from RecursiveArrayTools.jl, or a structured flat container such
as a `ComponentArray`. Boundary value problems have additional documented forms
for arrays of state guesses.

Container packages should define the relevant public
[ArrayInterface.jl](https://docs.sciml.ai/ArrayInterface/stable/) traits. In
particular, `ArrayInterface.ismutable` must agree with whether solver caches may
write into the value. If `ArrayInterface.fast_scalar_indexing` is `false`, the
container and selected algorithm need compatible bulk operations; declaring the
trait does not by itself make scalar-indexing solver code GPU-compatible.

### Mutable Containers

In-place models and mutable solver caches require writable values with
type-preserving operations. Depending on the algorithm, this includes:

  - `similar` for the same geometry and for requested element types or axes;
  - `copy`, `copyto!`, `fill!`, and `zero`;
  - fused and unfused broadcast assignment for state-state and scalar-state
    arithmetic; and
  - `getindex` and `setindex!` when the algorithm uses scalar indexing.

`resize!` is optional and is needed only for workflows that change the state
dimension through the integrator interface. Immutable containers such as
`SVector` instead use out-of-place model and cache paths. GPU arrays require an
algorithm, differentiation backend, callback set, and linear solver that all
avoid unsupported host scalar indexing.

## Matrix and Operator Types

The matrix or operator type used by an implicit algorithm need not match the
state container type. When a solver constructs a Jacobian from the state,
`ArrayInterface.zeromatrix(u)` should return a compatible matrix with the correct
dimensions and storage location. This is how recursive and GPU-aware containers
can select an appropriate default matrix representation.

The selected differentiation and linear-solver paths determine the remaining
contract. A concrete matrix may need matrix-vector products, `mul!`, compatible
`similar` and copy operations, diagonal access through `diagind`, and writable
diagonal views for shifts used to form a Newton matrix. A lazy operator should
implement the public
[SciMLOperators.jl](https://docs.sciml.ai/SciMLOperators/stable/) interface
instead of imitating a dense array.

Ultimately, the generated system `LinearProblem(A, b)` must be accepted by the
chosen linear solver, with `b` using the state-compatible right-hand-side type.
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
