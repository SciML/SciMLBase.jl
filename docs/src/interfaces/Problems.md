# SciMLProblems

The cornerstone of the SciML common interface is the problem type definition.
These definitions are the encoding of mathematical problems into a numerically
computable form. 

### Note About Symbolics and ModelingToolkit

The symbolic analog to the problem interface is the ModelingToolkit `AbstractSystem`.
For example, `ODESystem` is the symbolic analog to `ODEProblem`. Each of these system
types have a method for constructing the associated problem and function types.

## Definition of the SciMLProblem Interface

The following standard principles should be adhered to across all 
`SciMLProblem` instantiations.

### In-place Specification

Each `SciMLProblem` type can be called with an "is inplace" (iip) choice. For example:

```julia
ODEProblem(f,u0,tspan,p)
ODEProblem{iip}(f,u0,tspan,p)
```

which is a boolean for whether the function is in the inplace form (mutating to
change the first value). This is automatically determined using the methods table
but note that for full type-inferrability of the `SciMLProblem` this iip-ness should
be specified.

Additionally, the functions are fully specialized to reduce the runtimes. If one
would instead like to not specialize on the functions to reduce compile time,
then one can set `recompile` to false.

### Default Parameters

By default, `SciMLProblem` types use the `SciMLBase.NullParameters()` singleton to
define the absence of parameters by default. The reason is because this throws an
informative error if the parameter is used or accessed within the user's function,
for example, `p[1]` will throw an informative error about forgetting to pass
parameters.

### Keyword Argument Splatting

All `SciMLProblem` types allow for passing keyword arguments that would get forwarded
to the solver. The reason for this is that in many cases, like in `EnsembleProblem`
usage, a `SciMLProblem` might be associated with some solver configuration, such as a
callback or tolerance. Thus, for flexibility the extra keyword arguments to the
`SciMLProblem` are carried to the solver.

### problem_type

`SciMLProblem` types include a non-public API definition of `problem_type` which holds
a trait type corresponding to the way the `SciMLProblem` was constructed. For example,
if a `SecondOrderODEProblem` constructor is used, the returned problem is simply a
`ODEProblem` for interopability with any `ODEProblem` algorithm. However, in this case
the `problem_type` will be populated with the `SecondOrderODEProblem` type, indicating
the original definition and extra structure.

### Remake

```@docs
remake
```

## Problem Traits

```@docs
SciMLBase.isinplace(prob::SciMLBase.DEProblem)
SciMLBase.is_diagonal_noise
```

## SciMLProblem API

### Abstract SciMLProblems

```@docs
SciMLBase.SciMLProblem
SciMLBase.DEProblem
SciMLBase.AbstractLinearProblem
SciMLBase.AbstractNonlinearProblem
SciMLBase.AbstractQuadratureProblem
SciMLBase.AbstractOptimizationProblem
SciMLBase.AbstractNoiseProblem
SciMLBase.AbstractODEProblem
SciMLBase.AbstractDiscreteProblem
SciMLBase.AbstractAnalyticalProblem
SciMLBase.AbstractRODEProblem
SciMLBase.AbstractSDEProblem
SciMLBase.AbstractDAEProblem
SciMLBase.AbstractDDEProblem
SciMLBase.AbstractConstantLagDDEProblem
SciMLBase.AbstractSecondOrderODEProblem
SciMLBase.AbstractBVProblem
SciMLBase.AbstractJumpProblem
SciMLBase.AbstractSDDEProblem
SciMLBase.AbstractConstantLagSDDEProblem
SciMLBase.AbstractPDEProblem
```

### Concrete SciMLProblems

```@docs
LinearProblem
NonlinearProblem
QuadratureProblem
OptimizationProblem
BVProblem
DAEProblem
DDEProblem
DynamicalDDEProblem
SecondOrderDDEProblem
DiscreteProblem
NoiseProblem
ODEProblem
DynamicalODEProblem
SecondOrderODEProblem
SplitODEProblem
IncrementingODEProblem
RODEProblem
SDDEProblem
SplitSDEProblem
DynamicalSDEProblem
SteadyStateProblem
PDEProblem
```