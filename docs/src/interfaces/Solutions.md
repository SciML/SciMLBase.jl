# SciMLSolutions

## Definition of the SciMLSolution Interface

All `SciMLSolution` types are a subset of some `AbstractArray`. Types with time series
(like `ODESolution`) are subtypes of `RecursiveArrayTools.AbstractVectorOfArray` and 
`RecursiveArrayTools.AbstractDiffEqArray` where appropriate. Types without a time series
(like `OptimizationSolution`) are directly subsets of `AbstractArray`. 

### Array Interface

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

### Common Field Names

- `u`: the solution values
- `t`: the independent variable values, matching the length of the solution, if applicable
- `resid`: the residual of the solution, if applicable
- `original`: the solution object from the original solver, if it's a wrapper algorithm
- `retcode`: see the documentation section on return codes
- `prob`: the problem that was solved
- `alg`: the algorithm used to solve the problem

### [Return Codes (RetCodes)](@id retcodes)

The solution types have a `retcode` field which returns a symbol signifying the
error state of the solution. The retcodes are as follows:

- `:Default`: The solver did not set retcodes.
- `:Success`: The integration completed without erroring or the steady state solver
  from `SteadyStateDiffEq` found the steady state.
- `:Terminated`: The integration is terminated with `terminate!(integrator)`.
  Note that this may occur by using `TerminateSteadyState` from the callback
  library `DiffEqCallbacks`.
- `:MaxIters`: The integration exited early because it reached its maximum number
  of iterations.
- `:DtLessThanMin`: The timestep method chose a stepsize which is smaller than the
  allowed minimum timestep, and exited early.
- `:Unstable`: The solver detected that the solution was unstable and exited early.
- `:InitialFailure`: The DAE solver could not find consistent initial conditions.
- `:ConvergenceFailure`: The internal implicit solvers failed to converge.
- `:Failure`: General uncategorized failures or errors.

## Traits

## SciMLSolution API

### Abstract SciML Solutions

```@docs
SciMLBase.SciMLSolution
SciMLBase.AbstractNoTimeSolution
SciMLBase.AbstractTimeseriesSolution
SciMLBase.AbstractNoiseProcess
SciMLBase.AbstractEnsembleSolution
SciMLBase.AbstractLinearSolution
SciMLBase.AbstractNonlinearSolution
SciMLBase.AbstractQuadratureSolution
SciMLBase.AbstractSteadyStateSolution
SciMLBase.AbstractAnalyticalSolution
SciMLBase.AbstractODESolution
SciMLBase.AbstractDDESolution
SciMLBase.AbstractRODESolution
SciMLBase.AbstractDAESolution
```

### Concrete SciML Solutions

```@docs
SciMLBase.LinearSolution
SciMLBase.QuadratureSolution
SciMLBase.DAESolution
SciMLBase.NonlinearSolution
SciMLBase.ODESolution
SciMLBase.OptimizationSolution
SciMLBase.RODESolution
```