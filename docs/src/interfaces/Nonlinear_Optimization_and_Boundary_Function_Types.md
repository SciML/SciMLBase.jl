# [Nonlinear, Optimization, and Boundary Function Types](@id nonlinear_optimization_boundary_function_types)

These wrappers describe residual, objective, integrand, and boundary-condition
callbacks together with optional derivatives and structural metadata. The
in-place convention and each auxiliary callback signature are part of the
solver-facing interface.

## Nonlinear Functions

```@docs
SciMLBase.NonlinearFunction
SciMLBase.HomotopyNonlinearFunction
SciMLBase.IntervalNonlinearFunction
```

## Integral and Optimization Functions

```@docs
SciMLBase.IntegralFunction
SciMLBase.BatchIntegralFunction
SciMLBase.OptimizationFunction
SciMLBase.MultiObjectiveOptimizationFunction
```

## Boundary-Value Functions

```@docs
SciMLBase.BVPFunction
SciMLBase.TwoPointBVPFunction
SciMLBase.TwoPointDynamicalBVPFunction
SciMLBase.DynamicalBVPFunction
```
