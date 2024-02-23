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

Additionally, the functions are fully specialized to reduce the runtimes. If one
would instead like to not specialize on the functions to reduce compile time,
then one can set `recompile` to false.

### Specialization Levels

Specialization levels in problem definitions are used to control the amount of compilation
specialization is performed on the model functions in order to trade off between runtime
performance, simplicity, and compile-time performance. The default choice of specialization
is `AutoSpecialize`, which seeks to allow for using fully precompiled solvers in common
scenarios but falls back to a runtime-optimal approach when further customization is used.

Specialization levels are given as the second type parameter in `AbstractSciMLProblem`
constructors. For example, this is done via:

```julia
ODEProblem{iip, specialization}(f, u0, tspan, p)
```

Note that `iip` choice is required for specialization choices to be made.

#### Specialization Choices

```@docs
SciMLBase.AbstractSpecialization
SciMLBase.AutoSpecialize
SciMLBase.NoSpecialize
SciMLBase.FunctionWrapperSpecialize
SciMLBase.FullSpecialize
```

!!! note
    
    The specialization level must be precompile snooped in the appropriate solver
    package in order to enable the full precompilation and system image generation
    for zero-latency usage. By default, this is only done with AutoSpecialize and
    on types `u isa Vector{Float64}`, `eltype(tspan) isa Float64`, and
    `p isa Union{Vector{Float64}, SciMLBase.NullParameters}`. Precompilation snooping
    in the solvers can be done using the Preferences.jl setup on the appropriate
    solver. See the solver library's documentation for more details.

### Default Parameters

By default, `AbstractSciMLProblem` types use the `SciMLBase.NullParameters()` singleton to
define the absence of parameters by default. The reason is because this throws an
informative error if the parameter is used or accessed within the user's function,
for example, `p[1]` will throw an informative error about forgetting to pass
parameters.

### Keyword Argument Splatting

All `AbstractSciMLProblem` types allow for passing keyword arguments that would get forwarded
to the solver. The reason for this is that in many cases, like in `EnsembleProblem`
usage, a `AbstractSciMLProblem` might be associated with some solver configuration, such as a
callback or tolerance. Thus, for flexibility the extra keyword arguments to the
`AbstractSciMLProblem` are carried to the solver.

### problem_type

`AbstractSciMLProblem` types include a non-public API definition of `problem_type` which holds
a trait type corresponding to the way the `AbstractSciMLProblem` was constructed. For example,
if a `SecondOrderODEProblem` constructor is used, the returned problem is simply a
`ODEProblem` for interoperability with any `ODEProblem` algorithm. However, in this case
the `problem_type` will be populated with the `SecondOrderODEProblem` type, indicating
the original definition and extra structure.

### Remake

```@docs
remake
```

## Problem Traits

```@docs
SciMLBase.isinplace(prob::SciMLBase.AbstractDEProblem)
SciMLBase.is_diagonal_noise
```

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
SciMLBase.AbstractNonlinearProblem
SciMLBase.AbstractIntegralProblem
SciMLBase.AbstractOptimizationProblem
SciMLBase.AbstractNoiseProblem
SciMLBase.AbstractODEProblem
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
```
