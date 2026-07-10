# [Concrete Solution Types](@id concrete_solution_types)

Concrete solution types store solver results and implement the array and field
contracts described by the [SciML solution interface](@ref scimlsolutions).
Constructors for these types are primarily used by solver packages through
[`build_solution`](@ref SciMLBase.build_solution).

## Algebraic Solutions

```@docs
SciMLBase.LinearSolution
SciMLBase.EigenvalueSolution
SciMLBase.NonlinearSolution
SciMLBase.SteadyStateSolution
SciMLBase.IntegralSolution
SciMLBase.OptimizationSolution
```

## Differential Equation Solutions

```@docs
SciMLBase.ODESolution
SciMLBase.RODESolution
SciMLBase.DAESolution
```
