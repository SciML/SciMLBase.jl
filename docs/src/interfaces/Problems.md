# [SciMLProblems](@id scimlproblems)

The cornerstone of the SciML common interface is the problem type definition.
These definitions are the encoding of mathematical problems into a numerically
computable form.

### Note About Symbolics and ModelingToolkit

The symbolic analog to the problem interface is the ModelingToolkit `AbstractSystem`.
For example, `ODESystem` is the symbolic analog to `ODEProblem`. Each of these system
types have a method for constructing the associated problem and function types.

## Definition of the AbstractSciMLProblem Interface

The following standard principles should be adhered to across all
`AbstractSciMLProblem` instantiations.

### In-place Specification

Each `AbstractSciMLProblem` type can be called with an "is inplace" (iip) choice. For example:

```julia
ODEProblem(f, u0, tspan, p)
ODEProblem{iip}(f, u0, tspan, p)
```

which is a boolean for whether the function is in the inplace form (mutating to
change the first value). This is automatically determined using the methods table
but note that for full type-inferability of the `AbstractSciMLProblem` this iip-ness should
be specified.

By default, problem functions use `AutoSpecialize` to balance latency and runtime.
Choose another specialization marker explicitly when a workflow needs a different
trade-off.

### [Specialization Levels](@id specialization_levels)

Specialization levels in problem definitions are used to control the amount of compilation
specialization is performed on the model functions in order to trade off between runtime
performance, simplicity, and compile-time performance. The default choice of specialization
is `AutoSpecialize`, which seeks to allow for using fully precompiled solvers in common
scenarios but falls back to a runtime-optimal approach when further customization is used.

Specialization levels are given as the second explicit type parameter, after the
in-place flag, in concrete problem and function constructors. For example:

```julia
ODEProblem{iip, specialization}(f, u0, tspan, p)
```

Note that `iip` choice is required for specialization choices to be made.

#### Specialization Choices

```@docs
SciMLBase.AbstractSpecialization
SciMLBase.AutoSpecialize
SciMLBase.AutoDePSpecialize
SciMLBase.NoSpecialize
SciMLBase.FunctionWrapperSpecialize
SciMLBase.FullSpecialize
```

#### Specialization Interface Hooks

Solver and modeling packages should query the marker rather than inspect concrete
function type parameters. Packages that implement callable wrapping extend the
wrapper hooks below; ordinary user code should normally select a marker on the
problem constructor and let the selected solver perform any wrapping.

```@docs
SciMLBase.specialization
SciMLBase.isfunctionwrapper
SciMLBase.wrapfun_oop
SciMLBase.wrapfun_iip
SciMLBase.unwrap_fw
```

!!! note

    Precompiled solver methods can be reused only for signatures that the selected
    solver package precompiles. The covered state, time, parameter, and option types
    are solver-specific. Use `FullSpecialize` when a model falls outside the wrapped
    signatures supported by a solver or when runtime performance is the priority.

### Default Parameters

By default, `AbstractSciMLProblem` types use the `SciMLBase.NullParameters()` singleton to
define the absence of parameters by default. The reason is because this throws an
informative error if the parameter is used or accessed within the user's function,
for example, `p[1]` will throw an informative error about forgetting to pass
parameters.

```@docs
SciMLBase.NullParameters
```

### Keyword Argument Splatting

All `AbstractSciMLProblem` types allow for passing keyword arguments that would get forwarded
to the solver. The reason for this is that in many cases, like in `EnsembleProblem`
usage, a `AbstractSciMLProblem` might be associated with some solver configuration, such as a
callback or tolerance. Thus, for flexibility the extra keyword arguments to the
`AbstractSciMLProblem` are carried to the solver.

### Structured Constructors and Preservation

Convenience constructors may return a shared concrete problem representation while
preserving enough construction metadata for dispatch and `remake`. Downstream packages
must query `problem_type(prob)` instead of inspecting or mutating internal storage. A
convenience constructor's return type is therefore not by itself a complete description
of the mathematical structure used to create it.

### Remake

```@docs
SciMLBase.remake
```

For problems that are created from a system (e.g. created through ModelingToolkit.jl) or
define a DSL using `SymbolicIndexingInterface.SymbolCache`, `remake` can accept symbolic
maps as `u0` or `p`. A symbolic map is a `Dict` or `Vector{<:Pair}` mapping symbols in
`u0` or `p` to their values. These values can be numeric, or expressions of other symbols.
Symbolic maps can be complete (specifying a value for each symbol in `u0` or `p`) or
partial. For a partial symbolic map, the values of remaining symbols are obtained through
the system's defaults (see `SymbolicIndexingInterface.default_values`) and the existing
values in the problem passed to `remake`.

If the system's defaults contain an expression for the missing symbol, that expression
will be used for the value (it is treated as a dependent initialization). Otherwise,
the existing value of that symbol in the problem passed to `remake` is used.

If `use_defaults = true` is passed as a keyword argument to `remake`, then an available
numeric system default is preferred over the value already stored in the problem.

For example, consider a problem `prob` with parameters `:a`, `:b`, `:c` having values
`1.0`, `2.0`, `3.0` respectively. Let us also assume that the system contains the
defaults `Dict(:a => :(2b), :c => 0.1)`. Then:

  - `remake(prob; p = [:b => 2.0])` will result in the values `4.0`, `2.0`, `3.0` for
    `:a`, `:b` and `:c` respectively. Note how the numeric default for `:c` was not
    respected.
  - `remake(prob; p = [:b => 2.0], use_defaults = true)` will result in the values `4.0`,
    `2.0`, `0.1` for `:a`, `:b` and `:c` respectively.
  - `remake(prob; p = [:b => 2.0, :a => 3.0])` will result in the values `3.0`, `2.0`,
    `3.0` for `:a`, `:b` and `:c` respectively. Note how the explicitly specified value for
    `:a` overrides the dependent default.

### Aliasing Specification

An `AbstractAliasSpecifier` is associated with each SciML problem type that
allows solver caches to reuse problem inputs. See the
[alias specifier interface](@ref alias_specifier_interface) for the common
tri-state rules and the problem-family-specific specifiers.

## Problem Traits

Problem traits expose properties that are stored on concrete problem types and
used by solver dispatch. Solver packages should query these traits instead of
reconstructing the answer from fields or callback method tables. The detailed
contract is documented in [Problem Traits](@ref problem_traits).

## AbstractSciMLProblem API

### Defaults and Preferences

`SpecializationLevel` at `SciMLBase` can be used to set the default specialization level. The following
shows how to set the specialization default to `FullSpecialize`:

```julia
using Preferences, UUIDs
set_preferences!(
    UUID("0bca4576-84f4-4d90-8ffe-ffa030f20462"), "SpecializationLevel" => "FullSpecialize")
```

The default is `AutoSpecialize`.

### Abstract SciMLProblems

```@docs
SciMLBase.AbstractSciMLProblem
SciMLBase.AbstractDEProblem
SciMLBase.AbstractLinearProblem
SciMLBase.AbstractEigenvalueProblem
SciMLBase.AbstractNonlinearProblem
SciMLBase.AbstractIntervalNonlinearProblem
SciMLBase.AbstractIntegralProblem
SciMLBase.AbstractOptimizationProblem
SciMLBase.AbstractNoiseProblem
SciMLBase.AbstractODEProblem
SciMLBase.AbstractDynamicalODEProblem
SciMLBase.AbstractDynamicOptProblem
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
SciMLBase.AbstractSteadyStateProblem
```

### Problem Support Interfaces

```@docs
SciMLBase.AbstractOptimizationCache
```

## Concrete Problem Reference

Concrete constructors, their stored data, and the layout markers returned by
[`problem_type`](@ref SciMLBase.problem_type) are grouped by problem family:

  - [Algebraic Problem Types](@ref algebraic_problem_types)
  - [Differential Equation Problem Types](@ref differential_equation_problem_types)
  - [Delay, Boundary, and Noise Problem Types](@ref delay_boundary_noise_problem_types)

## Problem Utilities

```@docs
SciMLBase.promote_tspan
```
