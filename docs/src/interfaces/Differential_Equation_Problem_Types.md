# [Differential Equation Problem Types](@id differential_equation_problem_types)

These concrete problems store an evolution function, initial data, parameters,
an independent-variable span, and solver keyword arguments. Specialized
constructors may use a shared concrete representation; downstream code must use
[`problem_type`](@ref SciMLBase.problem_type) to recover the construction layout.

## Ordinary Differential Equations

```@docs
SciMLBase.ODEProblem
SciMLBase.ImmutableODEProblem
SciMLBase.StandardODEProblem
SciMLBase.DynamicalODEProblem
SciMLBase.SecondOrderODEProblem
SciMLBase.AbstractSplitODEProblem
SciMLBase.SplitODEProblem
SciMLBase.IncrementingODEProblem
```

## Discrete Equations

```@docs
SciMLBase.DiscreteProblem
SciMLBase.ImplicitDiscreteProblem
```

## Random and Stochastic Differential Equations

```@docs
SciMLBase.RODEProblem
SciMLBase.SDEProblem
SciMLBase.SplitSDEProblem
SciMLBase.DynamicalSDEProblem
```

## Differential-Algebraic Equations

```@docs
SciMLBase.DAEProblem
```
