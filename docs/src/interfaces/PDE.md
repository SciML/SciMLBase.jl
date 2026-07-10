# The PDE Definition Interface

PDE discretization packages bridge high-level equation descriptions to the
solver-ready problems used throughout the SciML ecosystem. The shared interface
is based on dispatch: a package extends [`discretize`](@ref SciMLBase.discretize)
for each supported pair of PDE representation and discretizer, and may also
extend [`symbolic_discretize`](@ref SciMLBase.symbolic_discretize) to expose the
intermediate representation.

SciMLBase does not prescribe one PDE equation language, domain type, boundary
condition syntax, or spatial discretization. Those semantics belong to the
packages that own the high-level representation and discretizer. This keeps the
result of discretization compatible with ordinary ODE, nonlinear,
optimization, linear, and other SciML solvers without requiring those solvers to
understand PDE-specific syntax.

## Core Extension Contract

A discretizer package implements:

```julia
SciMLBase.discretize(sys, discretizer, args...; kwargs...)
```

for each supported `sys` and `discretizer` pair. The method must return a
solver-ready SciML problem, such as an `ODEProblem`, `NonlinearProblem`, or
`OptimizationProblem`, or another documented problem type accepted by the
intended solver. It should forward relevant problem-construction keyword
arguments, validate unsupported equations, domains, and boundary conditions,
and preserve enough metadata to reconstruct the PDE solution.

[`AbstractDiscretization`](@ref SciMLBase.AbstractDiscretization) is an optional
common marker for discretizer algorithms. Implementing the interface does not
require subtyping it: a package may own a more specific public algorithm
hierarchy and still participate by extending `SciMLBase.discretize`.

A package may additionally implement:

```julia
SciMLBase.symbolic_discretize(sys, discretizer, args...; kwargs...)
```

The return type is part of the discretizer's public contract. It may be a
lowered symbolic system, a tuple containing a system and time span, a collection
of generated operators, or a diagnostic representation such as loss functions.
It need not be a ModelingToolkit `AbstractSystem`, but it should describe the
same discretization used by `discretize` and retain the information needed for
inspection or downstream transformations.

## PDE Representations

SciMLBase defines [`AbstractPDEProblem`](@ref SciMLBase.AbstractPDEProblem) and
the lightweight [`PDEProblem`](@ref SciMLBase.PDEProblem) wrapper. `PDEProblem`
stores a package-specific PDE representation together with extrapolation and
spatial metadata; SciMLBase does not interpret those fields.

High-level packages may instead define their own representation and dispatch
directly on it. For example, ModelingToolkit owns `PDESystem`, including its
equations, independent and dependent variables, parameters, domains, and
boundary or initial conditions. A downstream discretizer should document which
representation types it accepts rather than assuming every PDE enters through
`PDEProblem`.

```@docs
SciMLBase.PDEProblem
```

## Domains and Boundary Conditions

Domain and boundary-condition syntax is owned by the high-level PDE and
discretizer packages. SciMLBase does not require a universal interval, mesh,
geometry, or boundary-condition type. A discretizer's public interface should
document:

  - supported domain dimensions, geometries, coordinate systems, and grid or
    sampling specifications;
  - supported initial, Dirichlet, Neumann, Robin, periodic, interface, and other
    boundary-condition classes;
  - how conditions are validated, eliminated, embedded, or transformed; and
  - restrictions imposed by the selected discretization, including required
    regularity, compatible derivative orders, and unsupported combinations.

Invalid or unsupported input should produce an actionable error before the
generated numerical problem is passed to a solver.

## Discretization Functions

```@docs
SciMLBase.discretize
SciMLBase.symbolic_discretize
```

## Discretization Metadata and Solution Wrapping

Discretizers that return PDE-aware solutions can subtype
[`AbstractDiscretizationMetadata`](@ref SciMLBase.AbstractDiscretizationMetadata).
Use `AbstractDiscretizationMetadata{Val(true)}` when the generated solution has a
saved time axis and `AbstractDiscretizationMetadata{Val(false)}` when it does
not. [`wrap_sol(sol, metadata)`](@ref SciMLBase.wrap_sol) routes these to
`PDETimeSeriesSolution(sol, metadata)` and `PDENoTimeSolution(sol, metadata)`,
respectively.

The package that owns the metadata must define the corresponding outer solution
constructor. Its metadata should retain the original variables and domains, the
layout of generated state variables, and any grids, transformations, or
interpolants needed to map the solver output back to the PDE representation. If
the wrapper is callable, metadata-specific methods should define the supported
evaluation coordinates, interpolation behavior, and extrapolation behavior.

```@docs
SciMLBase.PDETimeSeriesSolution
SciMLBase.PDENoTimeSolution
SciMLBase.wrap_sol
```

## Transformations and Analysis

Equation simplification, index reduction, domain decomposition, coordinate
transformations, sparsity analysis, and operator construction belong to the
packages that own the PDE representation or discretizer. When such a
transformation changes variables or state layout, the discretizer must preserve
the map needed for solution reconstruction. `symbolic_discretize` is the common
inspection hook for exposing the lowered equations, operators, grids, loss
functions, or other package-defined intermediate data.

## Downstream Patterns

[MethodOfLines.jl](https://docs.sciml.ai/MethodOfLines/stable/) extends the
interface for `MOLFiniteDifference`. Its symbolic path returns a semi-discrete
system and time span; its solver-ready path constructs an `ODEProblem` for
time-dependent systems or a `NonlinearProblem` for stationary systems. Its
metadata records the discrete space and maps the numerical solution back to the
original PDE variables and grids.

[NeuralPDE.jl](https://docs.sciml.ai/NeuralPDE/stable/) extends the interface for
`PhysicsInformedNN`. Its symbolic path returns a representation containing the
generated loss functions and parameters, while `discretize` constructs an
`OptimizationProblem`. This demonstrates why `symbolic_discretize` has a
package-defined return type and why the shared contract is dispatch-based rather
than tied to a single discretizer hierarchy.
