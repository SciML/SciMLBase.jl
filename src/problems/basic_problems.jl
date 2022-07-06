@doc doc"""

Defines a linear system problem.
Documentation Page: http://linearsolve.sciml.ai/dev/basics/LinearProblem/

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
the [SciMLBase Documentation](https://scimlbase.sciml.ai/dev/).

Note that matrix-free versions of LinearProblem definitions are not compatible
with all solvers. To check a solver for compatibility, use the function xxxxx.

## Problem Type

### Constructors

Optionally, an initial guess ``u₀`` can be supplied which is used for iterative
methods.

```julia
LinearProblem{isinplace}(A,x,p=NullParameters();u0=nothing,kwargs...)
LinearProblem(f::AbstractDiffEqOperator,u0,p=NullParameters();u0=nothing,kwargs...)
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

@doc doc"""

Defines a nonlinear system problem.
Documentation Page: https://nonlinearsolve.sciml.ai/dev/basics/NonlinearProblem/

## Mathematical Specification of a Nonlinear Problem

To define a Nonlinear Problem, you simply need to give the function ``f``
which defines the nonlinear system:

```math
f(u,p) = 0
```

and an initial guess ``u₀`` of where `f(u,p)=0`. `f` should be specified as `f(u,p)`
(or in-place as `f(du,u,p)`), and `u₀` should be an AbstractArray (or number)
whose geometry matches the desired geometry of `u`. Note that we are not limited
to numbers or vectors for `u₀`; one is allowed to provide `u₀` as arbitrary
matrices / higher-dimension tensors as well.

## Problem Type

### Constructors

```julia
NonlinearProblem(f::NonlinearFunction,u0,p=NullParameters();kwargs...)
NonlinearProblem{isinplace}(f,u0,p=NullParameters();kwargs...)
```

`isinplace` optionally sets whether the function is in-place or not. This is
determined automatically, but not inferred.

Parameters are optional, and if not given, then a `NullParameters()` singleton
will be used, which will throw nice errors if you try to index non-existent
parameters. Any extra keyword arguments are passed on to the solvers. For example,
if you set a `callback` in the problem, then that `callback` will be added in
every solve call.

For specifying Jacobians and mass matrices, see the [NonlinearFunctions](@ref nonlinearfunctions)
page.

### Fields

* `f`: The function in the problem.
* `u0`: The initial guess for the steady state.
* `p`: The parameters for the problem. Defaults to `NullParameters`.
* `kwargs`: The keyword arguments passed on to the solvers.
"""
struct NonlinearProblem{uType, isinplace, P, F, K} <:
       AbstractNonlinearProblem{uType, isinplace}
    f::F
    u0::uType
    p::P
    kwargs::K
    @add_kwonly function NonlinearProblem{iip}(f::AbstractNonlinearFunction{iip}, u0,
                                               p = NullParameters(); kwargs...) where {iip}
        new{typeof(u0), iip, typeof(p), typeof(f), typeof(kwargs)}(f, u0, p, kwargs)
    end

    """
    $(SIGNATURES)

    Define a steady state problem using the given function.
    `isinplace` optionally sets whether the function is inplace or not.
    This is determined automatically, but not inferred.
    """
    function NonlinearProblem{iip}(f, u0, p = NullParameters()) where {iip}
        NonlinearProblem{iip}(NonlinearFunction{iip}(f), u0, p)
    end
end

"""
$(SIGNATURES)

Define a steady state problem using an instance of
[`AbstractNonlinearFunction`](@ref AbstractNonlinearFunction).
"""
function NonlinearProblem(f::AbstractNonlinearFunction, u0, p = NullParameters(); kwargs...)
    NonlinearProblem{isinplace(f)}(f, u0, p; kwargs...)
end

function NonlinearProblem(f, u0, p = NullParameters(); kwargs...)
    NonlinearProblem(NonlinearFunction(f), u0, p; kwargs...)
end

"""
$(SIGNATURES)

Define a steady state problem from a standard ODE problem.
"""
function NonlinearProblem(prob::AbstractNonlinearProblem)
    NonlinearProblem{isinplace(prob)}(prob.f, prob.u0, prob.p)
end

@doc doc"""

Defines an integral problem.
Documentation Page: https://github.com/SciML/Integrals.jl

## Mathematical Specification of a Integral Problem

Integral problems are multi-dimensional integrals defined as:

```math
\int_{lb}^{ub} f(u,p) du
```

where ``p`` are parameters. `u` is a `Number` or `AbstractArray`
whose geometry matches the space being integrated.

## Problem Type

### Constructors

IntegralProblem{iip}(f,lb,ub,p=NullParameters();
                  nout=1, batch = 0, kwargs...)

- f: the integrand, `dx=f(x,p)` for out-of-place or `f(dx,x,p)` for in-place.
- lb: Either a number or vector of lower bounds.
- ub: Either a number or vector of upper bounds.
- p: The parameters associated with the problem.
- nout: The output size of the function f. Defaults to 1, i.e., a scalar integral output.
- batch: The preferred number of points to batch. This allows user-side parallelization
  of the integrand. If batch != 0, then each x[:,i] is a different point of the integral
  to calculate, and the output should be nout x batchsize. Note that batch is a suggestion
  for the number of points, and it is not necessarily true that batch is the same as
  batchsize in all algorithms.
- kwargs:: Keyword arguments copied to the solvers.

Additionally, we can supply iip like IntegralProblem{iip}(...) as true or false to declare at
compile time whether the integrator function is in-place.

### Fields

The fields match the names of the constructor arguments.
"""
struct IntegralProblem{isinplace, P, F, L, U, K} <: AbstractIntegralProblem{isinplace}
    f::F
    lb::L
    ub::U
    nout::Int
    p::P
    batch::Int
    kwargs::K
    @add_kwonly function IntegralProblem{iip}(f, lb, ub, p = NullParameters();
                                              nout = 1,
                                              batch = 0, kwargs...) where {iip}
        new{iip, typeof(p), typeof(f), typeof(lb),
            typeof(ub), typeof(kwargs)}(f, lb, ub, nout, p, batch, kwargs)
    end
end

function IntegralProblem(f, lb, ub, args...; kwargs...)
    IntegralProblem{isinplace(f, 3)}(f, lb, ub, args...; kwargs...)
end

struct QuadratureProblem end
@deprecate QuadratureProblem(args...; kwargs...) IntegralProblem(args...; kwargs...)

@doc doc"""

Defines a optimization problem.
Documentation Page: https://galacticoptim.sciml.ai/dev/API/optimization_problem/

## Mathematical Specification of a Optimization Problem

To define an Optimization Problem, you simply need to give the function ``f``
which defines the cost function to minimize:

```math
min_u f(u,p)
```

``u₀`` is an initial guess of the minimum. `f` should be specified as `f(u,p)`
and `u₀` should be an AbstractArray (or number) whose geometry matches the
desired geometry of `u`. Note that we are not limited to numbers or vectors
for `u₀`; one is allowed to provide `u₀` as arbitrary matrices /
higher-dimension tensors as well.

## Problem Type

### Constructors

```julia
OptimizationProblem{iip}(f, x, p = SciMLBase.NullParameters(),;
                        lb = nothing,
                        ub = nothing,
                        lcons = nothing,
                        ucons = nothing,
                        sense = nothing,
                        kwargs...)
```

`isinplace` optionally sets whether the function is in-place or not. This is
determined automatically, but not inferred. Note that for OptimizationProblem,
in-place only refers to the Jacobian and Hessian functions, and thus by default
if the `OptimizationFunction` is not defined directly then `iip = true` is
done by default.

Parameters are optional, and if not given, then a `NullParameters()` singleton
will be used, which will throw nice errors if you try to index non-existent
parameters. Any extra keyword arguments are passed on to the solvers. For example,
if you set a `callback` in the problem, then that `callback` will be added in
every solve call.

`lb` and `ub` are the upper and lower bounds for box constraints on the
optimization. They should be an `AbstractArray` matching the geometry of `u`,
where `(lb[I],ub[I])` is the box constraint (lower and upper bounds)
for `u[I]`.

`lcons` and `ucons` are the upper and lower bounds for equality constraints on the
optimization. They should be an `AbstractArray` matching the geometry of `u`,
where `(lcons[I],ucons[I])` is the constraint (lower and upper bounds)
for `cons[I]`.

If `f` is a standard Julia function, it is automatically converted into an
`OptimizationFunction` with `NoAD()`, i.e., no automatic generation
of the derivative functions.

Any extra keyword arguments are captured to be sent to the optimizers.

### Fields

* `f`: the function in the problem.
* `u0`: the initial guess for the optima.
* `p`: the parameters for the problem. Defaults to `NullParameters`.
* `lb`: the lower bounds for the optimization of `u`.
* `ub`: the upper bounds for the optimization of `u`.
* `lcons`: the vector of lower bounds for the constraints passed to [OptimizationFunction](@ref).
    Defaults to `nothing`, implying no lower bounds for the constraints (i.e. the constraint bound is `-Inf`)
* `ucons`: the vector of upper bounds for the constraints passed to [OptimizationFunction](@ref).
    Defaults to `nothing`, implying no upper bounds for the constraints (i.e. the constraint bound is `Inf`)
* `sense`: the objective sense, can take `MaxSense` or `MinSense` from Optimization.jl.
* `kwargs`: the keyword arguments passed on to the solvers.

## Inequality and Equality Constraints

Both inequality and equality constraints are defined by the `f.cons` function in the `OptimizationFunction`
description of the problem structure. This `f.cons` is given as a function `f.cons(u,p)` which computes
the value at `u` of the constraints. For example, take `f.cons(u,p) = u[1] - u[2]` 
With these definitions, `lcons` and `ucons` define the bounds on the constraint that the solvers try to satisfy.
If `lcons` and `ucons` are `nothing`, then there are no constraints bounds, meaning that the constraint is satisfied when `-Inf < f.cons < Inf` (which of course is always!). If `lcons[i] = ucons[i] = 0`, then the constraint is satisfied when `f.cons(u,p)[i] = 0`, and so this implies the equality constraint ``u[1] = u[2]`. If `lcons[i] = ucons[i] = a`, then ``u[1] - u[2] = a`` is the equality constraint. 

Inequality constraints are then given by making `lcons[i] != ucons[i]`. For example, `lcons[i] = -Inf` and `ucons[i] = 0` would imply the inequality constraint ``u[1] < u[2]`` since any `f.cons[i] < 0` satisfies the constraint. Similarly, `lcons[i] = -1` and `ucons[i] = 1` would imply that `-1 < f.cons[i] < 1` is required or ``-1 < u[1] - u[2] < 1``.

Note that these vectors must be sized to match the number of constraints, with one set of conditions for each constraint.

"""
struct OptimizationProblem{iip, F, uType, P, B, LC, UC, S, K} <:
       AbstractOptimizationProblem{iip}
    f::F
    u0::uType
    p::P
    lb::B
    ub::B
    lcons::LC
    ucons::UC
    sense::S
    kwargs::K
    @add_kwonly function OptimizationProblem{iip}(f::OptimizationFunction{iip}, u0,
                                                  p = NullParameters();
                                                  lb = nothing, ub = nothing,
                                                  lcons = nothing, ucons = nothing,
                                                  sense = nothing, kwargs...) where {iip}
        if xor(lb === nothing, ub === nothing)
            error("If any of `lb` or `ub` is provided, both must be provided.")
        end
        new{iip, typeof(f), typeof(u0), typeof(p),
            typeof(lb), typeof(lcons), typeof(ucons),
            typeof(sense), typeof(kwargs)}(f, u0, p, lb, ub, lcons, ucons, sense, kwargs)
    end
end

function OptimizationProblem(f::OptimizationFunction, args...; kwargs...)
    OptimizationProblem{isinplace(f)}(f, args...; kwargs...)
end
function OptimizationProblem(f, args...; kwargs...)
    OptimizationProblem{true}(OptimizationFunction{true}(f), args...; kwargs...)
end

isinplace(f::OptimizationFunction{iip}) where {iip} = iip
isinplace(f::OptimizationProblem{iip}) where {iip} = iip
