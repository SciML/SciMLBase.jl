# [Delay, Boundary, and Noise Problem Types](@id delay_boundary_noise_problem_types)

Delay problems add a history function and lag data to the common evolution
problem contract. Boundary-value problems add residual callbacks evaluated over
the candidate solution or at boundary points. Noise problems request generation
or replay of a noise-process trajectory.

## Delay Differential Equations

```@docs
SciMLBase.DDEProblem
SciMLBase.StandardDDEProblem
SciMLBase.AbstractDynamicalDDEProblem
SciMLBase.DynamicalDDEProblem
SciMLBase.SecondOrderDDEProblem
SciMLBase.SDDEProblem
```

## Boundary-Value Problems

```@docs
SciMLBase.BVProblem
SciMLBase.StandardBVProblem
SciMLBase.TwoPointBVProblem
SciMLBase.SecondOrderBVProblem
SciMLBase.StandardSecondOrderBVProblem
SciMLBase.TwoPointSecondOrderBVProblem
```

## Noise Problems

```@docs
SciMLBase.NoiseProblem
```
