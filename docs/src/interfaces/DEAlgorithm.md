# The DEAlgorithm Interface

`DEAlgorithm` is the high level type for differential equation solver
algorithms defined by SciMLBase.jl. DiffEqBase.jl has a high level
algorithm for handling preprocessing of differential equation solves,
and relies on the correctly-defined traits of the constituant solvers
in order to make correct determinations about the behavior. The interface
traits are:

```@docs
isautodifferentiable
allows_arbitrary_number_types
allowscomplex
isadaptive
isdiscrete
```