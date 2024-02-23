# SciMLAlgorithms

## Definition of the AbstractSciMLAlgorithm Interface

`SciMLAlgorithms` are defined as types which have dispatches to the function signature:

```julia
CommonSolve.solve(prob::AbstractSciMLProblem, alg::AbstractSciMLAlgorithm; kwargs...)
```

### Algorithm-Specific Arguments

Note that because the keyword arguments of `solve` are designed to be common across the whole
problem type, algorithms should have the algorithm-specific keyword arguments defined as part
of the algorithm constructor. For example, `Rodas5` has a choice of `autodiff::Bool` which is
not common across all ODE solvers, and thus `autodiff` is a algorithm-specific keyword argument
handled via `Rodas5(autodiff=true)`.

### Remake

Note that `remake` is applicable to `AbstractSciMLAlgorithm` types, but this is not used in the public API.
It's used for solvers to swap out components like ForwardDiff chunk sizes.

## Common Algorithm Keyword Arguments

Commonly used algorithm keyword arguments are:

## Traits

```@docs
SciMLBase.isautodifferentiable
SciMLBase.allows_arbitrary_number_types
SciMLBase.allowscomplex
SciMLBase.isadaptive
SciMLBase.isdiscrete
SciMLBase.forwarddiffs_model
SciMLBase.forwarddiffs_model_time
```

### Abstract SciML Algorithms

```@docs
SciMLBase.AbstractSciMLAlgorithm
SciMLBase.AbstractDEAlgorithm
SciMLBase.AbstractLinearAlgorithm
SciMLBase.AbstractNonlinearAlgorithm
SciMLBase.AbstractIntervalNonlinearAlgorithm
SciMLBase.AbstractQuadratureAlgorithm
SciMLBase.AbstractOptimizationAlgorithm
SciMLBase.AbstractSteadyStateAlgorithm
SciMLBase.AbstractODEAlgorithm
SciMLBase.AbstractSecondOrderODEAlgorithm
SciMLBase.AbstractRODEAlgorithm
SciMLBase.AbstractSDEAlgorithm
SciMLBase.AbstractDAEAlgorithm
SciMLBase.AbstractDDEAlgorithm
SciMLBase.AbstractSDDEAlgorithm
```
