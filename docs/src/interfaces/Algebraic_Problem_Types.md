# [Algebraic Problem Types](@id algebraic_problem_types)

These concrete problems cover algebraic equations, quadrature, optimization,
steady states, and analytically evaluated trajectories. Their constructors
normalize user functions into the corresponding SciML function wrappers and
preserve solver keyword arguments for `solve`.

## Linear and Eigenvalue Problems

```@docs
SciMLBase.LinearProblem
SciMLBase.EigenvalueProblem
SciMLBase.EigenvalueTarget
SciMLBase.EigenvalueTarget.LargestMagnitude
SciMLBase.EigenvalueTarget.SmallestMagnitude
SciMLBase.EigenvalueTarget.LargestRealPart
SciMLBase.EigenvalueTarget.SmallestRealPart
SciMLBase.EigenvalueTarget.LargestImaginaryPart
SciMLBase.EigenvalueTarget.SmallestImaginaryPart
```

## Nonlinear Problems

```@docs
SciMLBase.NonlinearProblem
SciMLBase.StandardNonlinearProblem
SciMLBase.IntervalNonlinearProblem
SciMLBase.NonlinearLeastSquaresProblem
SciMLBase.SCCNonlinearProblem
SciMLBase.HomotopyProblem
```

## Integral and Optimization Problems

```@docs
SciMLBase.IntegralProblem
SciMLBase.SampledIntegralProblem
SciMLBase.OptimizationProblem
```

## Steady-State and Analytical Problems

```@docs
SciMLBase.SteadyStateProblem
SciMLBase.AnalyticalProblem
```
