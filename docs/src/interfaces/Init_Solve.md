# The SciML init and solve Functions

`solve` function has the default definition

```julia
solve(args...; kwargs...) = solve!(init(args...; kwargs...))
```

The interface for the three functions is as follows:

```julia
init(::ProblemType, args...; kwargs...) :: IteratorType
solve!(::SolverType) :: SolutionType
```

where `ProblemType`, `IteratorType`, and `SolutionType` are the types defined in
your package.

To avoid method ambiguity, the first argument of `solve`, `solve!`, and `init`
_must_ be dispatched on the type defined in your package.  For example, do
_not_ define a method such as

```julia
init(::AbstractVector, ::AlgorithmType)
```

## `init` and the Iterator Interface

`init`'s return gives an `IteratorType` which is designed to allow the user to
have more direct handling over the internal solving process. Because of this
internal nature, the `IteratorType` has a less unified interface across problem
types than other portions like `ProblemType` and `SolutionType`. For example,
for differential equations this is the 
[Integrator Interface](https://diffeq.sciml.ai/stable/basics/integrator/)
designed for mutating solutions in a manner for callback implementation, which
is distinctly different from the 
[LinearSolve init interface](http://linearsolve.sciml.ai/dev/tutorials/caching_interface/)
which is designed for caching efficiency with reusing factorizations.