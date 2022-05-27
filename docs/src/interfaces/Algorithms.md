# SciMLAlgorithms

## Definition of the SciMLAlgorithm Interface

`SciMLAlgorithms` are defined as types which have dispatches to the function signature:

```julia
CommonSolve.solve(prob::SciMLProblem,alg::SciMLAlgorithm;kwargs...)
```

### Algorithm-Specific Arguments

Note that because the keyword arguments of `solve` are designed to be common across the whole
problem type, algorithms should have the algorithm-specific keyword arguments defined as part
of the algorithm constructor. For example, `Rodas5` has a choice of `autodiff::Bool` which is
not common across all ODE solvers, and thus `autodiff` is a algorithm-specific keyword argument
handled via `Rodas5(autodiff=true)`.

### Remake

Note that `remake` is applicable to `SciMLAlgorithm` types, but this is not used in the public API.
It's used for solvers to swap out components like ForwardDiff chunk sizes.

## Common Algorithm Keyword Arguments

Commonly used algorithm keyword arguments are:

## Traits

```@docs
isautodifferentiable
allows_arbitrary_number_types
allowscomplex
isadaptive
isdiscrete
```

### Abstract SciML Algorithms

```@docs
SciMLAlgorithm
DEAlgorithm
AbstractLinearAlgorithm
AbstractNonlinearAlgorithm
AbstractQuadratureAlgorithm
AbstractSteadyStateAlgorithm
AbstractODEAlgorithm
AbstractSecondOrderODEAlgorithm
AbstractRODEAlgorithm
AbstractSDEAlgorithm
AbstractDAEAlgorithm
AbstractDDEAlgorithm
AbstractSDDEAlgorithm
```

### Concrete SciML Algorithms

The concrete SciML algorithms are found in the respective solver documentations.