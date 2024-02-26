@enum ObjSense MinSense MaxSense

@doc doc"""

Defines an optimization problem.
Documentation Page: https://docs.sciml.ai/Optimization/stable/API/optimization_problem/

## Mathematical Specification of an Optimization Problem

To define an optimization problem, you need the objective function ``f``
which is minimized over the domain of ``u``, the collection of optimization variables:

```math
min_u f(u,p)
```

``u₀`` is an initial guess for the minimizer. `f` should be specified as `f(u,p)`
and `u₀` should be an `AbstractArray` whose geometry matches the
desired geometry of `u`. Note that we are not limited to vectors
for `u₀`; one is allowed to provide `u₀` as arbitrary matrices /
higher-dimension tensors as well.

## Problem Type

### Constructors

```julia
OptimizationProblem{iip}(f, u0, p = SciMLBase.NullParameters(),;
                        lb = nothing,
                        ub = nothing,
                        lcons = nothing,
                        ucons = nothing,
                        sense = nothing,
                        kwargs...)
```

`isinplace` optionally sets whether the function is in-place or not.
This is determined automatically, but not inferred. Note that for OptimizationProblem,
in-place refers to the objective's derivative functions, the constraint function
and its derivatives. `OptimizationProblem` currently only supports in-place.

Parameters `p` are optional, and if not given, then a `NullParameters()` singleton
will be used, which will throw nice errors if you try to index non-existent
parameters.

`lb` and `ub` are the upper and lower bounds for box constraints on the
optimization variables. They should be an `AbstractArray` matching the geometry of `u`,
where `(lb[i],ub[i])` is the box constraint (lower and upper bounds) for `u[i]`.

`lcons` and `ucons` are the upper and lower bounds in case of inequality constraints on the
optimization and if they are set to be equal then it represents an equality constraint.
They should be an `AbstractArray` matching the geometry of `u`, where `(lcons[i],ucons[i])`
are the lower and upper bounds for `cons[i]`.

The `f` in the `OptimizationProblem` should typically be an instance of [`OptimizationFunction`](https://docs.sciml.ai/Optimization/stable/API/optimization_function/#optfunction)
to specify the objective function and its derivatives either by passing
predefined functions for them or automatically generated using the [ADType](https://github.com/SciML/ADTypes.jl).

If `f` is a standard Julia function, it is automatically transformed into an
`OptimizationFunction` with `NoAD()`, meaning the derivative functions are not
automatically generated.

Any extra keyword arguments are captured to be sent to the optimizers.

### Fields

* `f`: the function in the problem.
* `u0`: the initial guess for the optimization variables.
* `p`: the constant parameters used for defining the problem. Defaults to `NullParameters`.
* `lb`: the lower bounds for the optimization variables `u`.
* `ub`: the upper bounds for the optimization variables `u`.
* `int`: integrality indicator for `u`. If `int[i] == true`, then `u[i]` is an integer variable.
    Defaults to `nothing`, implying no integrality constraints.
* `lcons`: the vector of lower bounds for the constraints passed to [OptimizationFunction](https://docs.sciml.ai/Optimization/stable/API/optimization_function/#optfunction).
    Defaults to `nothing`, implying no lower bounds for the constraints (i.e. the constraint bound is `-Inf`)
* `ucons`: the vector of upper bounds for the constraints passed to [`OptimizationFunction`](https://docs.sciml.ai/Optimization/stable/API/optimization_function/#optfunction).
    Defaults to `nothing`, implying no upper bounds for the constraints (i.e. the constraint bound is `Inf`)
* `sense`: the objective sense, can take `MaxSense` or `MinSense` from Optimization.jl.
* `kwargs`: the keyword arguments passed on to the solvers.

## Inequality and Equality Constraints

Both inequality and equality constraints are defined by the `f.cons` function in the [`OptimizationFunction`](https://docs.sciml.ai/Optimization/stable/API/optimization_function/#optfunction)
description of the problem structure. This `f.cons` is given as a function `f.cons(u,p)` which computes
the value of the constraints at `u`. For example, take `f.cons(u,p) = u[1] - u[2]`.
With these definitions, `lcons` and `ucons` define the bounds on the constraint that the solvers try to satisfy.
If `lcons` and `ucons` are `nothing`, then there are no constraints bounds, meaning that the constraint is satisfied when `-Inf < f.cons < Inf` (which of course is always!). If `lcons[i] = ucons[i] = 0`, then the constraint is satisfied when `f.cons(u,p)[i] = 0`, and so this implies the equality constraint `u[1] = u[2]`. If `lcons[i] = ucons[i] = a`, then ``u[1] - u[2] = a`` is the equality constraint.

Inequality constraints are then given by making `lcons[i] != ucons[i]`. For example, `lcons[i] = -Inf` and `ucons[i] = 0` would imply the inequality constraint ``u[1] <= u[2]`` since any `f.cons[i] <= 0` satisfies the constraint. Similarly, `lcons[i] = -1` and `ucons[i] = 1` would imply that `-1 <= f.cons[i] <= 1` is required or ``-1 <= u[1] - u[2] <= 1``.

Note that these vectors must be sized to match the number of constraints, with one set of conditions for each constraint.

"""
struct OptimizationProblem{iip, F, uType, P, LB, UB, I, LC, UC, S, K} <:
       AbstractOptimizationProblem{iip}
    f::F
    u0::uType
    p::P
    lb::LB
    ub::UB
    int::I
    lcons::LC
    ucons::UC
    sense::S
    kwargs::K
    @add_kwonly function OptimizationProblem{iip}(f::OptimizationFunction{iip}, u0,
            p = NullParameters();
            lb = nothing, ub = nothing, int = nothing,
            lcons = nothing, ucons = nothing,
            sense = nothing, kwargs...) where {iip}
        if xor(lb === nothing, ub === nothing)
            error("If any of `lb` or `ub` is provided, both must be provided.")
        end
        warn_paramtype(p)
        new{iip, typeof(f), typeof(u0), typeof(p),
            typeof(lb), typeof(ub), typeof(int), typeof(lcons), typeof(ucons),
            typeof(sense), typeof(kwargs)}(f, u0, p, lb, ub, int, lcons, ucons, sense,
            kwargs)
    end
end

function OptimizationProblem(f::OptimizationFunction, args...; kwargs...)
    OptimizationProblem{isinplace(f)}(f, args...; kwargs...)
end
function OptimizationProblem(f, args...; kwargs...)
    isinplace(f, 2, has_two_dispatches = false)
    OptimizationProblem{true}(OptimizationFunction{true}(f), args...; kwargs...)
end

function OptimizationFunction(
        f::NonlinearFunction, adtype::AbstractADType = NoAD(); kwargs...)
    if isinplace(f)
        throw(ArgumentError("Converting NonlinearFunction to OptimizationFunction is not supported with in-place functions yet."))
    end
    OptimizationFunction((u, p) -> sum(abs2, f(u, p)), adtype; kwargs...)
end

function OptimizationProblem(
        prob::NonlinearLeastSquaresProblem, adtype::AbstractADType = NoAD(); kwargs...)
    if isinplace(prob)
        throw(ArgumentError("Converting NonlinearLeastSquaresProblem to OptimizationProblem is not supported with in-place functions yet."))
    end
    optf = OptimizationFunction(prob.f, adtype; kwargs...)
    return OptimizationProblem(optf, prob.u0, prob.p; prob.kwargs..., kwargs...)
end

isinplace(f::OptimizationFunction{iip}) where {iip} = iip
isinplace(f::OptimizationProblem{iip}) where {iip} = iip
