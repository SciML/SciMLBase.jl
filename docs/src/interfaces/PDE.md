# The PDE Definition Interface

While ODEs ``u' = f(u,p,t)`` can be defined by a user-function `f`, for PDEs the
function form can be different for every PDE. How many functions, and how many
inputs? This can always change. The SciML ecosystem solves this problem by
using [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/) to define `PDESystem`,
a high-level symbolic description of the PDE to be consumed by other packages.

The vision for the common PDE interface is that a user should only have to specify
their PDE once, mathematically, and have instant access to everything as simple
as a finite difference method with constant grid spacing, to something as complex
as a distributed multi-GPU discontinuous Galerkin method.

The key to the common PDE interface is a separation of the symbolic handling from
the numerical world. All of the discretizers should not "solve" the PDE, but
instead be a conversion of the mathematical specification to a numerical problem.
Preferably, the transformation should be to another ModelingToolkit.jl `AbstractSystem`
via a `symbolic_discretize` dispatch, but in some cases this cannot be done or will
not be performant. Thus in some cases, only a `discretize` definition is given to a
`AbstractSciMLProblem`, with `symbolic_discretize` simply providing diagnostic or lower level
information about the construction process.

These elementary problems, such as solving linear systems `Ax=b`, solving nonlinear
systems `f(x)=0`, ODEs, etc. are all defined by SciMLBase.jl, which then numerical
solvers can all target these common forms. Thus someone who works on linear solvers
doesn't necessarily need to be working on a Discontinuous Galerkin or finite element
library, but instead "linear solvers that are good for matrices A with
properties ..." which are then accessible by every other discretization method
in the common PDE interface.

Similar to the rest of the `AbstractSystem` types, transformation and analyses
functions will allow for simplifying the PDE before solving it, and constructing
block symbolic functions like Jacobians.

## Problem and Discretization Contracts

The common PDE interface has three layers:

  - an [`AbstractPDEProblem`](@ref SciMLBase.AbstractPDEProblem) or concrete
    [`PDEProblem`](@ref SciMLBase.PDEProblem) stores the pre-discretization PDE
    representation and domain metadata;
  - an [`AbstractDiscretization`](@ref SciMLBase.AbstractDiscretization)
    describes the numerical transformation from the high-level PDE
    representation to a solver-ready SciML problem; and
  - an [`AbstractDiscretizationMetadata`](@ref SciMLBase.AbstractDiscretizationMetadata)
    value records how the generated numerical solution maps back to the original
    independent variables, dependent variables, and domains.

Discretization metadata also determines how the final solver solution is wrapped.
Use `AbstractDiscretizationMetadata{Val(true)}` when the generated PDE solution
has a saved time axis and `AbstractDiscretizationMetadata{Val(false)}` when it
does not. The generic `SciMLBase.wrap_sol(sol, metadata)` dispatch uses that flag
to call either `PDETimeSeriesSolution(sol, metadata)` or
`PDENoTimeSolution(sol, metadata)`. Downstream discretizer packages should define
the corresponding outer constructor for their metadata type and should keep
callable interpolation/evaluation methods in the package that owns the metadata.

## Constructors

Symbolic PDE systems are constructed with `ModelingToolkit.PDESystem`, which is
owned by ModelingToolkit.jl.

```@docs
SciMLBase.PDEProblem
```

### Domains (WIP)

Domains are specifying by saying `indepvar in domain`, where `indepvar` is a
single or a collection of independent variables, and `domain` is the chosen
domain type. A 2-tuple can be used to indicate an `Interval`.
Thus forms for the `indepvar` can be like:

```julia
t ∈ (0.0, 1.0)
(t, x) ∈ UnitDisk()
[v, w, x, y, z] ∈ VectorUnitBall(5)
```

#### Domain Types (WIP)

  - `Interval(a,b)`: Defines the domain of an interval from `a` to `b` (requires explicit
    import from `DomainSets.jl`, but a 2-tuple can be used instead)

## `discretize` and `symbolic_discretize`

The only functions which act on a PDESystem are the following:

  - `discretize(sys,discretizer)`: produces the outputted `AbstractSystem` or
    `AbstractSciMLProblem`.
  - `symbolic_discretize(sys,discretizer)`: produces a debugging symbolic description
    of the discretized problem.

```@docs
SciMLBase.discretize
SciMLBase.symbolic_discretize
```

## PDE Solution Wrappers

```@docs
SciMLBase.PDETimeSeriesSolution
SciMLBase.PDENoTimeSolution
SciMLBase.wrap_sol
```

## Boundary Conditions (WIP)

## Transformations

## Analyses

## Discretizer Ecosystem

### NeuralPDE.jl: PhysicsInformedNN

[NeuralPDE.jl](https://docs.sciml.ai/NeuralPDE/stable/) defines the `PhysicsInformedNN`
discretizer which uses a [DiffEqFlux.jl](https://docs.sciml.ai/DiffEqFlux/stable/)
neural network to solve the differential equation.

### MethodOfLines.jl: MOLFiniteDifference (WIP)

[MethodOfLines.jl](https://docs.sciml.ai/MethodOfLines/stable/) defines the
`MOLFiniteDifference` discretizer which performs a finite difference discretization
using the DiffEqOperators.jl stencils. These stencils make use of NNLib.jl for
fast operations on semi-linear domains.
