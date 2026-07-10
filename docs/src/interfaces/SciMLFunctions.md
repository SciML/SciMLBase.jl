# [SciMLFunctions (Jacobians, Sparsity, Etc.)](@id scimlfunctions)

The SciML ecosystem provides an extensive interface for declaring extra functions
associated with the differential equation's data. In traditional libraries there
is usually only one option: the Jacobian. However, we allow for a large array
of pre-computed functions to speed up the calculations. This is offered via the
`SciMLFunction` types which can be passed to the problems.

## Definition of the AbstractSciMLFunction Interface

The following standard principles should be adhered to across all
`AbstractSciMLFunction` instantiations.

### Common Function Choice Definitions

The full interface available to the solvers is as follows:

  - `jac`: The Jacobian of the differential equation with respect to the state
    variable `u` at a time `t` with parameters `p`.
  - `paramjac`: The Jacobian of the differential equation with respect to `p` at
    state `u` at time `t`.
  - `analytic`: Defines an analytical solution using `u0` at time `t` with `p`
    which will cause the solvers to return errors. Used for testing.
  - `syms`: Allows you to name your variables for automatic names in plots and
    other output.
  - `jac_prototype`: Defines the type to be used for any internal Jacobians
    within the solvers.
  - `sparsity`: Defines the sparsity pattern to be used for the sparse differentiation
    schemes. By default this is equal to `jac_prototype`. See the sparsity handling
    portion of this page for more information.
  - `colorvec`: The coloring pattern used by the sparse differentiator. See the
    sparsity handling portion of this page for more information.
  - `observed`: A function which allows for generating other observables from a
    solution.

Each function type additionally has some specific arguments, refer to their
documentation for details.

### In-place Specification and No-Recompile Mode

Each SciMLFunction type can be called with an "is inplace" (iip) choice.

```julia
ODEFunction(f)
ODEFunction{iip}(f)
```

which is a boolean for whether the function is in the inplace form (mutating to
change the first value). This is automatically determined using the methods table
but note that for full type-inferability of the `AbstractSciMLProblem` this iip-ness should
be specified.

### Specialization Choices

Each `SciMLFunction` type allows for specialization choices

```julia
ODEFunction{iip, specialization}(f)
```

which designates how the compiler should specialize on the model function `f`. For
more details on specialization choices, see the [SciMLProblems](@ref scimlproblems)
page.

### Specifying Jacobian Types

The `jac` field of an inplace style `SciMLFunction` has the signature `jac(J,u,p,t)`,
which updates the Jacobian `J` in-place. The intended type for `J` can sometimes be
inferred (e.g. when it is just a dense `Matrix`), but not in general. To supply the
type information, you can provide a `jac_prototype` in the function's constructor.

The following example creates an inplace `ODEFunction` whose Jacobian is a `Diagonal`:

```julia
using LinearAlgebra
f = (du, u, p, t) -> du .= t .* u
jac = (J, u, p, t) -> (J[1, 1] = t; J[2, 2] = t; J)
jp = Diagonal(zeros(2))
fun = ODEFunction(f; jac = jac, jac_prototype = jp)
```

The prototype declares Jacobian shape, element type, and structure. It must support
the operations required by the selected differentiation and linear solver, including
writes for an in-place `jac` and any multiplication, diagonal-shift, or factorization
operations that solver performs. Solvers may copy, allocate with `similar`, or convert
the prototype, so code must not rely on its object identity or aliasing behavior.

When `jac_prototype` is an `AbstractSciMLOperator` and `jac` is omitted, the constructor
creates a Jacobian update using `update_coefficients!` for the in-place form and
`update_coefficients` for the out-of-place form.
Refer to the [SciMLOperators](https://docs.sciml.ai/SciMLOperators/stable/premade_operators/) section for more information
on setting up time/parameter dependent operators.

### Sparsity Handling

Solver packages select differentiation backends through their algorithm and option
interfaces. Depending on that choice, they may use finite differences, automatic
differentiation, sparse coloring, explicit derivative functions, or matrix-free
Jacobian-vector and vector-Jacobian products. The function wrapper provides structural
information without requiring a particular backend package.

Set `jac_prototype` to the concrete matrix or operator representation that should be
used for the Jacobian. Set `sparsity` to the structural nonzero pattern used for sparse
differentiation; it defaults to `jac_prototype` when omitted. Set `colorvec` to a
column-color vector compatible with both that pattern and the selected backend, or
leave it as `nothing` so coloring can be computed when needed. A prototype and color
vector are performance and representation contracts, not substitutes for operations
required by the selected linear solver.

### Function Wrapper Rules

SciMLBase provides small wrapper types that turn an ODE-style model function
into a one-variable callable for derivative code. These wrappers close over the
fixed arguments and expose only the argument being differentiated.

- Time wrappers fix `u` and `p`, then expose `t`.
- State wrappers fix `t` and `p`, then expose `u`.
- `isinplace(wrapper)` preserves the in-place convention detected from the
  wrapped function.
- In-place wrappers support caller-provided output arrays. Their one-argument
  convenience calls allocate an output with `similar` to match the exposed state.
- Wrapper trait queries forward to the underlying function where applicable, so
  derivative code should query traits instead of inspecting wrapper fields.

## Traits

```@docs
SciMLBase.isinplace
SciMLBase.unwrapped_f
SciMLBase.has_analytic
SciMLBase.has_jac
SciMLBase.has_jvp
SciMLBase.has_vjp
SciMLBase.has_tgrad
SciMLBase.has_initialization_data
```

## AbstractSciMLFunction API

### Abstract SciML Functions

```@docs
SciMLBase.AbstractSciMLFunction
SciMLBase.AbstractDiffEqFunction
SciMLBase.AbstractODEFunction
SciMLBase.AbstractSDEFunction
SciMLBase.AbstractDDEFunction
SciMLBase.AbstractDAEFunction
SciMLBase.AbstractRODEFunction
SciMLBase.AbstractDiscreteFunction
SciMLBase.AbstractSDDEFunction
SciMLBase.AbstractNonlinearFunction
SciMLBase.AbstractIntervalNonlinearFunction
SciMLBase.AbstractIntegralFunction
SciMLBase.AbstractOptimizationFunction
SciMLBase.AbstractODEInputFunction
SciMLBase.AbstractBVPFunction
SciMLBase.AbstractParameterizedFunction
SciMLBase.AbstractHistoryFunction
```

### Concrete SciML Functions

```@docs
SciMLBase.ODEFunction
```

### Automatic Differentiation Markers

```@docs
SciMLBase.NoAD
```

### Function Wrappers

```@docs
SciMLBase.TimeDerivativeWrapper
SciMLBase.TimeGradientWrapper
SciMLBase.UDerivativeWrapper
SciMLBase.UJacobianWrapper
SciMLBase.IncrementingODEFunction
```
