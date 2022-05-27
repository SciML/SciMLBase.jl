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
but note that for full type-inferrability of the `SciMLProblem` this iip-ness should
be specified.

Additionally, the functions are fully specialized to reduce the runtimes. If one
would instead like to not specialize on the functions to reduce compile time,
then one can set `recompile` to false.

```julia
ODEFunction{iip,false}(f)
```

This makes the ODE solver compilation independent of the function and so changing
the function will not cause recompilation. One can change the default value
by changing the `const RECOMPILE_BY_DEFAULT = true` to false in the SciMLBase.jl
source code.

### Specifying Jacobian Types

The `jac` field of an inplace style `SciMLFunction` has the signature `jac(J,u,p,t)`,
which updates the jacobian `J` in-place. The intended type for `J` can sometimes be
inferred (e.g. when it is just a dense `Matrix`), but not in general. To supply the
type information, you can provide a `jac_prototype` in the function's constructor.

The following example creates an inplace `ODEFunction` whose jacobian is a `Diagonal`:

```julia
using LinearAlgebra
f = (du,u,p,t) -> du .= t .* u
jac = (J,u,p,t) -> (J[1,1] = t; J[2,2] = t; J)
jp = Diagonal(zeros(2))
fun = ODEFunction(f; jac=jac, jac_prototype=jp)
```

Note that the integrators will always make a deep copy of `fun.jac_prototype`, so
there's no worry of aliasing.

In general the jacobian prototype can be anything that has `mul!` defined, in
particular sparse matrices or custom lazy types that support `mul!`. A special case
is when the `jac_prototype` is a `AbstractDiffEqLinearOperator`, in which case you
do not need to supply `jac` as it is automatically set to `update_coefficients!`.
Refer to the [DiffEqOperators](@ref) section for more information
on setting up time/parameter dependent operators.

### Sparsity Handling

The solver libraries internally use packages such as [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl)
and [SparseDiffTools.jl](https://github.com/JuliaDiff/SparseDiffTools.jl) for
high performance calculation of sparse Jacobians and Hessians, along with matrix-free
calculations of Jacobian-Vector products (J*v), vector-Jacobian products (v'*J),
and Hessian-vector products (H*v). The SciML interface gives users the ability
to control these connections in order to allow for top notch performance.

The key arguments in the SciMLFunction is the `prototype`, which is an object
that will be used as the underlying Jacobian/Hessian. Thus if one wants to use
a sparse Jacobian, one should specify `jac_prototype` to be a sparse matrix.
The sparsity pattern used in the differentiation scheme is defined by `sparsity`.
By default, `sparsity=jac_prototype`, meaning that the sparse automatic differentiation
scheme should specialize on the sparsity pattern given by the actual sparsity
pattern. This can be overridden to say perform partial matrix coloring approximations.
Additionally, the color vector for the sparse differentiation directions can
be specified directly via `colorvec`. For more information on how these arguments
control the differentiation process, see the aforementioned differentiation
library documentations.

## Traits

```@docs
SciMLBase.isinplace(f::SciMLBase.AbstractSciMLFunction)
```

## AbstractSciMLFunction API

### Abstract SciML Functions

```@docs
SciMLBase.AbstractDiffEqFunction
SciMLBase.AbstractODEFunction
SciMLBase.AbstractSDEFunction
SciMLBase.AbstractDDEFunction
SciMLBase.AbstractDAEFunction
SciMLBase.AbstractRODEFunction
SciMLBase.AbstractDiscreteFunction
SciMLBase.AbstractSDDEFunction
SciMLBase.AbstractNonlinearFunction
```

### Concrete SciML Functions

```@docs
ODEFunction
SplitFunction
DynamicalODEFunction
DDEFunction
DynamicalDDEFunction
DiscreteFunction
SDEFunction
SplitSDEFunction
DynamicalSDEFunction
RODEFunction
DAEFunction
SDDEFunction
NonlinearFunction
OptimizationFunction
```