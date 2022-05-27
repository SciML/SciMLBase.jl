# SciMLSolutions

## Definition of the SciMLSolution Interface

All `SciMLSolution` types are a subset of some `AbstractArray`. Types with time series
(like `ODESolution`) are subtypes of `RecursiveArrayTools.AbstractVectorOfArray` and 
`RecursiveArrayTools.AbstractDiffEqArray` where appropriate. Types without a time series
(like `OptimizationSolution`) are directly subsets of `AbstractArray`. 

## Array Interface

Instead of working on the `Vector{uType}` directly, we can use the provided
array interface.

```julia
sol[j]
```

to access the value at timestep `j` (if the timeseries was saved), and

```julia
sol.t[j]
```

to access the value of `t` at timestep `j`. For multi-dimensional systems, this
will address first by component and lastly by time, and thus

```julia
sol[i,j]
```

will be the `i`th component at timestep `j`. Hence, `sol[j][i] == sol[i, j]`. This is done because Julia is column-major, so the leading dimension should be contiguous in memory. If the independent variables had shape
(for example, was a matrix), then `i` is the linear index. We can also access
solutions with shape:

```julia
sol[i,k,j]
```

gives the `[i,k]` component of the system at timestep `j`. The colon operator is
supported, meaning that

```julia
sol[i,:]
```

gives the timeseries for the `i`th component.

## Common Field Names

- `u`: the solution values
- `t`: the independent variable values, matching the length of the solution, if applicable
- `resid`: the residual of the solution, if applicable
- `original`: the solution object from the original solver, if it's a wrapper algorithm
- `retcode`: see the documentation section on return codes
- `prob`: the problem that was solved
- `alg`: the algorithm used to solve the problem

## Traits

## SciMLSolution API

### Abstract SciML Solutions

```@docs
SciMLSolution
DESolution
AbstractNoTimeSolution
AbstractTimeseriesSolution
AbstractNoiseProcess
AbstractEnsembleSolution
AbstractLinearSolution
AbstractNonlinearSolution
AbstractQuadratureSolution
AbstractSteadyStateSolution
AbstractAnalyticalSolution
AbstractODESolution
AbstractDDESolution
AbstractRODESolution
AbstractDAESolution
```

### Concrete SciML Solutions

```@docs
LinearSolution
QuadratureSolution
DAESolution
NonlinearSolution
SteadyStateSolution
ODESolution
OptimizationSolution
RODESolution
```