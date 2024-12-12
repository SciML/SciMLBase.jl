@doc doc"""

Defines a linear system problem.
Documentation Page: [https://docs.sciml.ai/LinearSolve/stable/basics/LinearProblem/](https://docs.sciml.ai/LinearSolve/stable/basics/LinearProblem/)

## Mathematical Specification of a Linear Problem

### Concrete LinearProblem

To define a `LinearProblem`, you simply need to give the `AbstractMatrix` ``A``
and an `AbstractVector` ``b`` which defines the linear system:

```math
Au = b
```

### Matrix-Free LinearProblem

For matrix-free versions, the specification of the problem is given by an
operator `A(u,p,t)` which computes `A*u`, or in-place as `A(du,u,p,t)`. These
are specified via the `AbstractSciMLOperator` interface. For more details, see
the [SciMLBase Documentation](https://docs.sciml.ai/SciMLBase/stable/).

Note that matrix-free versions of LinearProblem definitions are not compatible
with all solvers. To check a solver for compatibility, use the function xxxxx.

## Problem Type

### Constructors

Optionally, an initial guess ``u₀`` can be supplied which is used for iterative
methods.

```julia
LinearProblem{isinplace}(A,x,p=NullParameters();u0=nothing,kwargs...)
LinearProblem(f::AbstractSciMLOperator,u0,p=NullParameters();u0=nothing,kwargs...)
```

`isinplace` optionally sets whether the function is in-place or not, i.e. whether
the solvers are allowed to mutate. By default this is true for `AbstractMatrix`,
and for `AbstractSciMLOperator`s it matches the choice of the operator definition.

Parameters are optional, and if not given, then a `NullParameters()` singleton
will be used, which will throw nice errors if you try to index non-existent
parameters. Any extra keyword arguments are passed on to the solvers.

### Fields

* `A`: The representation of the linear operator.
* `b`: The right-hand side of the linear system.
* `p`: The parameters for the problem. Defaults to `NullParameters`. Currently unused.
* `u0`: The initial condition used by iterative solvers.
* `kwargs`: The keyword arguments passed on to the solvers.
"""
struct LinearProblem{uType, isinplace, F, bType, P, K} <:
       AbstractLinearProblem{bType, isinplace}
    A::F
    b::bType
    u0::uType
    p::P
    kwargs::K
    @add_kwonly function LinearProblem{iip}(A, b, p = NullParameters(); u0 = nothing,
            kwargs...) where {iip}
        warn_paramtype(p)
        new{typeof(u0), iip, typeof(A), typeof(b), typeof(p), typeof(kwargs)}(A, b, u0, p,
            kwargs)
    end
end

function LinearProblem(A, b, args...; kwargs...)
    if A isa AbstractArray
        LinearProblem{true}(A, b, args...; kwargs...)
    elseif A isa Number
        LinearProblem{false}(A, b, args...; kwargs...)
    else
        LinearProblem{isinplace(A, 4)}(A, b, args...; kwargs...)
    end
end

struct LinearAliasSpecifier <: AbstractAliasSpecifier
    alias_A::Union{Bool,Nothing}
    alias_b::Union{Bool,Nothing}
end

@doc doc"""
Holds information on what variables to alias
when solving a LinearProblem. Conforms to the AbstractAliasSpecifier interface. 
    `LinearAliasSpecifier(;alias_p = nothing, alias_f = nothing, alias_A = nothing, alias_b = nothing, alias = nothing)`

When a keyword argument is `nothing`, the default behaviour of the solver is used.

### Keywords

* `alias_p::Union{Bool, Nothing}`
* `alias_f::Union{Bool, Nothing}`
* `alias_A::Union{Bool, Nothing}`: alias the `A` array.
* `alias_b::Union{Bool, Nothing}`: alias the `b` array. 
* `alias::Union{Bool, Nothing}`: sets all fields of the `LinearAliasSpecifier` to `alias`. 

Creates a `LinearAliasSpecifier` where `alias_A` and `alias_b` default to `nothing`.
When `alias_A` or `alias_b` is nothing, the default value of the solver is used.
"""
function LinearAliasSpecifier(;alias_A = nothing, alias_b = nothing, alias_p = nothing, alias_f = nothing, alias = nothing)
    if alias == true 
        LinearAliasSpecifier(true,true,true,true)
    elseif alias == false
        LinearAliasSpecifier(false,false,false,false)
    elseif isnothing(alias)
        LinearAliasSpecifier(alias_p, alias_f, alias_A, alias_b)
    end
end