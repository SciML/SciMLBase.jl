# [ODE and Discrete Function Types](@id ode_discrete_function_types)

These wrappers define callback signatures and optional derivative, symbolic,
initialization, and structural metadata for ordinary and discrete evolution
problems. Solver implementations should query the documented traits rather than
inspect wrapper fields.

## Ordinary Differential Equation Functions

```@docs
SciMLBase.ODEFunction
SciMLBase.DynamicalODEFunction
SciMLBase.SplitFunction
SciMLBase.IncrementingODEFunction
SciMLBase.ODEInputFunction
```

## Discrete Functions

```@docs
SciMLBase.DiscreteFunction
SciMLBase.ImplicitDiscreteFunction
```
