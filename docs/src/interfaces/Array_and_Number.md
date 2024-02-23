# [SciML Container (Array) and Number Interfaces](@id arrayandnumber)

We live in a society, and therefore there are rules. In this tutorial we outline
the rules which are required on container and number types which are allowable
in SciML tools.

!!! warn
    
    In general as of 2023, strict adherence to this interface is an early work-in-progress.
    If anything does not conform to the documented interface, please open an issue.

!!! note
    
    There are many types which can work with a specific solver that do satisfy this
    interface. Many times as part of prototyping you may want to side-step the
    high level interface checks in order to simply test whether a new type is working.
    To do this, set `interface_checks = false` as a keyword argument to `init`/`solve`
    to bypass any of the internal interface checks. This means you will no longer get
    a nice high-level error message and instead it will attempt to use the type
    without restrictions. Note that not every problem/solver has implemented this
    new keyword argument as of 2023.

## Note About Wrapped Solvers

Due to limitations of wrapped solvers, any solver that is a wrapped solver from an existing C/Fortran
code is inherently limited to `Float64` and `Vector{Float64}` for its operations. This includes packages
like Sundials.jl, LSODA.jl, DASKR.jl, MINPACK.jl, and many more. This is fundamental to these solvers
and it is not expected that they will allow the full set of SciML types in the future. If more abstract
number/container definitions are required, then these are not the appropriate solvers to use.

## SciML Number Types

The number types are the types used to define the dependent variables (i.e. `u0`) and the
independent variables (`t` or `tspan`). These two types can be different, and can have
different restrictions depending on the type of solver which is employed. The following
rules for a Number type are held in general:

  - Number types can be used in SciML directly or in containers. If a problem defines a value like `u0`
    using a Number type, the out-of-place form must be used for the problem definition.
  - `x::T + y::T = z::T`
  - `x::T * y::T = z::T`
  - `oneunit(x::T)::T`
  - `one(x::T) * oneunit(x::T) = z::T`
  - `t::T2 * x::T + y::T = z::T` for `T2` a time type and `T` the dependent variable type (this includes the
    `muladd` equivalent form).

Additionally, the following rules apply to subsets of uses:

### Adaptive Number Types

  - `x::T / y::T = z::T`
  - Default choices of norms can assume `sqrt(x::T)::T` exists. If `internalnorm` is overridden then this
    may not be required (for example, changing the norm to inf-norm).
  - `x::T ^ y::T = z::T`

### Time Types (Independent Variables)

  - If a solver is time adaptive, the time type must be a floating point number. `Rational` is only allowed
    for non-adaptive solves.

## SciML Container (Array) Types

Container types are types which hold number types. They can be used to define objects like the state vector
(`u0`) of a problem. The following operations are required in a container type to be used with SciML
solvers:

  - Broadcast is defined [according to the Julia broadcast interface](https://docs.julialang.org/en/v1/manual/interfaces/#man-interfaces-broadcasting).
  - The container type correctly defines [interface overloads to satisfy the ArrayInterface.jl specification](https://docs.sciml.ai/ArrayInterface/stable/).
  - `ArrayInterface.zeromatrix(x::T)::T2` defines a compatible matrix type (see below)
  - `eltype(x::T)::T2` is a compatible Number type.
  - `x::T .+ y::T = z::T` (i.e. broadcast similar is defined to be type-presurving)
  - Indexing is only required if `ArrayInterface.fast_scalar_indexing(x::T)==true`. If true,
    scalar indexing `x[i]` is assumed to be defined and run through all variables.

!!! note
    
    "`eltype(x::T)::T2` is a compatible Number type" excludes `Array{Array{T}}` types of types. However, recursive
    vectors can conformed to the interface with zero overhead using tools from RecursiveArrayTools.jl such as
    `VectorOfArray(x)`. Since this greatly simplifies the interfaces and the ability to check for correctness,
    doing this wrapping is highly recommended and there are no plans to relax this requirement.

Additionally, the following rules apply to subsets of uses:

### SciML Mutable Array Types

  - `similar(x::T)::T`
  - `zero(x::T)::T`
  - `z::T .= x::T .+ y::T` is defined
  - `z::T .= x::T .* y::T` is defined
  - `z::T .= t::T2 .* x::T` where `T2` is the time type (a Number) and `T` is the container type.
  - (Optional) `Base.resize!(x,i)` is required for `resize!(integrator,i)` to be supported.

### SciML Matrix (Operator) Type

Note that the matrix type may not match the type of the initial container `u0`. An example is `ComponentMatrix`
as the matrix structure corresponding to a `ComponentArray`. However, the following actions are assumed
to hold on the resulting matrix type:

  - `solve(LinearProblem(A::T,b::T2),linsolve)` must be defined for a solver to work on a given SciML matrix
    type `T2`.
  - If the matrix is an operator, i.e. a lazy construct, it should conform to the
    [SciMLOperators](https://docs.sciml.ai/SciMLOperators/stable/) interface.
  - If not a SciMLOperator, `diagind(W::T)` should be defined and `@view(A[idxs])=@view(A[idxs]) + λ::T`
