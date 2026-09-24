# Reexported API

Not every name that `using SciMLBase` puts in scope is defined by SciMLBase. This page
lists the ones that are not, so it is always clear which package actually owns a name
and where its reference documentation lives.

## SciMLOperators

`using SciMLBase` also brings in the
[SciMLOperators.jl](https://docs.sciml.ai/SciMLOperators/stable/) operator interface, so
the operator-valued arguments of the SciML interface can be built without importing
SciMLOperators separately. Those arguments are documented throughout these pages: a
`jac_prototype` may be an `AbstractSciMLOperator`, in which case the Jacobian update goes
through `update_coefficients!`/`update_coefficients` (see
[SciMLFunctions](@ref scimlfunctions)); the linear half of a `SplitFunction` or
`SplitODEProblem`, and the function of an `IncrementingODEProblem`, are operators; and a
lazy matrix supplied as a SciML matrix type is expected to conform to the SciMLOperators
interface rather than imitate a dense array.

These names are **owned and documented by SciMLOperators**, not by SciMLBase. SciMLBase
only re-exports them; the reference documentation for each is upstream, under
[Premade Operators](https://docs.sciml.ai/SciMLOperators/stable/premade_operators/) and
[The SciMLOperators Interface](https://docs.sciml.ai/SciMLOperators/stable/interface/).

### Module binding

  - `SciMLOperators` — the module itself, so anything it owns can be reached qualified as
    `SciMLOperators.name` without importing SciMLOperators separately.

### Operator types

  - Trivial operators: `IdentityOperator`, `NullOperator`
  - Matrix-backed operators: `MatrixOperator`, `DiagonalOperator`, `AffineOperator`,
    `AddVector`
  - Scalar operators: `ScalarOperator`
  - Matrix-free operators: `FunctionOperator`
  - Structured and lazy compositions: `BlockDiagonalOperator`, `TensorProductOperator`,
    `TensorSumOperator`, `InvertibleOperator`

### Operator interface and traits

  - Updating state- and parameter-dependent coefficients: `update_coefficients`,
    `update_coefficients!`
  - Caching and materialization: `cache_operator`, `iscached`, `concretize`,
    `isconvertible`, `has_concretization`
  - Structural traits: `isconstant`, `islinear`, `issquare`
  - Supported-operation traits: `has_adjoint`, `has_mul`, `has_mul!`, `has_ldiv`,
    `has_ldiv!`, `has_exp`, `has_expmv`, `has_expmv!`
  - Lazy algebra helpers: `kronsum`

### Boundary

Anything else from SciMLOperators must be imported from SciMLOperators directly. Two
cases are worth calling out, because they are deliberate and would otherwise read as
omissions:

  - **`AbstractSciMLOperator`** — the abstract type these pages refer to — and the other
    abstract types and lazy-algebra result types (`AbstractSciMLScalarOperator`,
    `ScaledOperator`, `AddedOperator`, `ComposedOperator`, ...) are declared public but
    deliberately not exported by SciMLOperators. They are therefore not exported here
    either, and are used qualified: `SciMLOperators.AbstractSciMLOperator`, which the
    re-exported `SciMLOperators` binding above makes reachable from `using SciMLBase`.
  - **`WOperator`, `StaticWOperator`, `jacobian_stale`, `mark_jacobian_updated!` and
    `mark_jacobian_current!`** are the stiff-solver `W = M - γJ` machinery. They are
    constructed by solver packages such as OrdinaryDiffEq and consumed by LinearSolve,
    both of which depend on SciMLOperators directly, and are not part of the interface a
    `using SciMLBase` user writes against.

## CommonSolve

The solve verbs `solve`, `solve!`, `init` and `step!` are owned by
[CommonSolve.jl](https://github.com/SciML/CommonSolve.jl). SciMLBase defines the SciML
methods on them and re-exports them, since they are the SciML solve interface as users and
solver packages write it. See [The SciML init and solve Functions](@ref) for the SciML contract on
these functions.
