# SciMLSolutions

## Definition of the AbstractSciMLSolution Interface

All `AbstractSciMLSolution` types are a subset of some `AbstractArray`. Types with time series
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
sol[i, j]
```

will be the `i`th component at timestep `j`. Hence, `sol[j][i] == sol[i, j]`. This is done because Julia is column-major, so the leading dimension should be contiguous in memory. If the independent variables had shape
(for example, was a matrix), then `i` is the linear index. We can also access
solutions with shape:

```julia
sol[i, k, j]
```

gives the `[i,k]` component of the system at timestep `j`. The colon operator is
supported, meaning that

```julia
sol[i, :]
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

## [Return Codes (RetCodes)](@id retcodes)

The solution types have a `retcode` field which returns a `SciMLBase.ReturnCode.T`
(from [EnumX.jl](https://github.com/fredrikekre/EnumX.jl), see that package for the
semantics of handling EnumX types) signifying the error or satisfaction state of
the solution.

```@docs
SciMLBase.ReturnCode
```

### Return Code Traits

```@docs
SciMLBase.successful_retcode
```

### Specific Return Codes

```@docs
SciMLBase.ReturnCode.Default
SciMLBase.ReturnCode.Success
SciMLBase.ReturnCode.Terminated
SciMLBase.ReturnCode.DtNaN
SciMLBase.ReturnCode.MaxIters
SciMLBase.ReturnCode.DtLessThanMin
SciMLBase.ReturnCode.Unstable
SciMLBase.ReturnCode.InitialFailure
SciMLBase.ReturnCode.ConvergenceFailure
SciMLBase.ReturnCode.Failure
SciMLBase.ReturnCode.ExactSolutionLeft
SciMLBase.ReturnCode.ExactSolutionRight
SciMLBase.ReturnCode.FloatingPointLimit
```

## Solution Traits

## AbstractSciMLSolution API

### Abstract SciML Solutions

```@docs
SciMLBase.AbstractSciMLSolution
SciMLBase.AbstractNoTimeSolution
SciMLBase.AbstractTimeseriesSolution
SciMLBase.AbstractNoiseProcess
SciMLBase.AbstractEnsembleSolution
SciMLBase.AbstractLinearSolution
SciMLBase.AbstractNonlinearSolution
SciMLBase.AbstractIntegralSolution
SciMLBase.AbstractSteadyStateSolution
SciMLBase.AbstractAnalyticalSolution
SciMLBase.AbstractODESolution
SciMLBase.AbstractDDESolution
SciMLBase.AbstractRODESolution
SciMLBase.AbstractDAESolution
```
