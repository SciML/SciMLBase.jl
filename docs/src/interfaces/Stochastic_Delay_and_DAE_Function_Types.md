# [Stochastic, Delay, and DAE Function Types](@id stochastic_delay_dae_function_types)

These wrappers extend the evolution-function contract with noise functions,
history arguments, or derivative-state inputs. Their docstrings specify the
in-place and out-of-place callback signatures consumed by solver packages.

## Stochastic and Random Differential Equation Functions

```@docs
SciMLBase.SDEFunction
SciMLBase.SplitSDEFunction
SciMLBase.DynamicalSDEFunction
SciMLBase.RODEFunction
```

## Delay Differential Equation Functions

```@docs
SciMLBase.DDEFunction
SciMLBase.DynamicalDDEFunction
SciMLBase.SDDEFunction
```

## Differential-Algebraic Equation Functions

```@docs
SciMLBase.DAEFunction
```
