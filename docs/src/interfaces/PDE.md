# The PDE Definition Interface

While ODEs ``u' = f(u,p,t)`` can be defined by a user-function `f`, for PDEs the
function form can be different for every PDE. How many functions, and how many
inputs? This can always change. The SciML ecosystem solves this problem by
using [ModelingToolkit.jl](https://mtk.sciml.ai/dev/) to define `PDESystem`,
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
`SciMLProblem`, with `symbolic_discretize` simply providing diagnostic or lower level
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

## Constructors

```@docs
ModelingToolkit.PDESystem
```

### Domains (WIP)

Domains are specifying by saying `indepvar in domain`, where `indepvar` is a
single or a collection of independent variables, and `domain` is the chosen
domain type. A 2-tuple can be used to indicate an `Interval`.
Thus forms for the `indepvar` can be like:

```julia
t ∈ (0.0,1.0)
(t,x) ∈ UnitDisk()
[v,w,x,y,z] ∈ VectorUnitBall(5)
```

#### Domain Types (WIP)

- `Interval(a,b)`: Defines the domain of an interval from `a` to `b` (requires explicit
import from `DomainSets.jl`, but a 2-tuple can be used instead)

## `discretize` and `symbolic_discretize`

The only functions which act on a PDESystem are the following:

- `discretize(sys,discretizer)`: produces the outputted `AbstractSystem` or
  `SciMLProblem`.
- `symbolic_discretize(sys,discretizer)`: produces a debugging symbolic description
  of the discretized problem.

## Boundary Conditions (WIP)

## Transformations

## Analyses

## Discretizer Ecosystem

### NeuralPDE.jl: PhysicsInformedNN

[NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl) defines the `PhysicsInformedNN`
discretizer which uses a [DiffEqFlux.jl](https://github.com/SciML/DiffEqFlux.jl)
neural network to solve the differential equation.

### MethodOfLines.jl: MOLFiniteDifference (WIP)

[MethodOfLines.jl](https://github.com/SciML/MethodOfLines.jl) defines the
`MOLFiniteDifference` discretizer which performs a finite difference discretization
using the DiffEqOperators.jl stencils. These stencils make use of NNLib.jl for
fast operations on semi-linear domains.
