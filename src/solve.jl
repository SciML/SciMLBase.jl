const NONCONCRETE_ELTYPE_MESSAGE = """
                                   Non-concrete element type inside of an `Array` detected.
                                   Arrays with non-concrete element types, such as
                                   `Array{Union{Float32,Float64}}`, are not supported by the
                                   differential equation solvers. Anyways, this is bad for
                                   performance so you don't want to be doing this!

                                   If this was a mistake, promote the element types to be
                                   all the same. If this was intentional, for example,
                                   using Unitful.jl with different unit values, then use
                                   an array type which has fast broadcast support for
                                   heterogeneous values such as the ArrayPartition
                                   from RecursiveArrayTools.jl. For example:

                                   ```julia
                                   using RecursiveArrayTools
                                   x = ArrayPartition([1.0,2.0],[1f0,2f0])
                                   y = ArrayPartition([3.0,4.0],[3f0,4f0])
                                   x .+ y # fast, stable, and usable as u0 into DiffEq!
                                   ```

                                   Element type:
                                   """

struct NonConcreteEltypeError <: Exception
    eltype::Any
end

function Base.showerror(io::IO, e::NonConcreteEltypeError)
    print(io, NONCONCRETE_ELTYPE_MESSAGE)
    print(io, e.eltype)
    println(io, TruncatedStacktraces.VERBOSE_MSG)
end

# Skip the DiffEqBase handling

struct IncompatibleOptimizerError <: Exception
    err::String
end

function Base.showerror(io::IO, e::IncompatibleOptimizerError)
    print(io, e.err)
end

"""
```julia
solve(prob::OptimizationProblem, alg::AbstractOptimizationAlgorithm,
    args...; kwargs...)::OptimizationSolution
```

For information about the returned solution object, refer to the documentation for [OptimizationSolution](@ref)

## Keyword Arguments

The arguments to `solve` are common across all of the optimizers.
These common arguments are:

  - `maxiters`: the maximum number of iterations
  - `maxtime`: the maximum amount of time (typically in seconds) the optimization runs for
  - `abstol`: absolute tolerance in changes of the objective value
  - `reltol`: relative tolerance  in changes of the objective value
  - `callback`: a callback function

Some optimizer algorithms have special keyword arguments documented in the
solver portion of the documentation and their respective documentation.
These arguments can be passed as `kwargs...` to `solve`. Similarly, the special
keyword arguments for the `local_method` of a global optimizer are passed as a
`NamedTuple` to `local_options`.

Over time, we hope to cover more of these keyword arguments under the common interface.

A warning will be shown if a common argument is not implemented for an optimizer.

## Callback Functions

The callback function `callback` is a function that is called after every optimizer
step. Its signature is:

```julia
callback = (state, loss_val) -> false
```

where `state` is an `OptimizationState` and stores information for the current
iteration of the solver and `loss_val` is loss/objective value. For more
information about the fields of the `state` look at the `OptimizationState`
documentation. The callback should return a Boolean value, and the default
should be `false`, so the optimization stops if it returns `true`.

### Callback Example

Here we show an example of a callback function that plots the prediction at the current value of the optimization variables.
For a visualization callback, we would need the prediction at the current parameters i.e. the solution of the `ODEProblem` `prob`.
So we call the `predict` function within the callback again.

```julia
function predict(u)
    Array(solve(prob, Tsit5(), p = u))
end

function loss(u, p)
    pred = predict(u)
    sum(abs2, batch .- pred)
end

callback = function (state, l; doplot = false) #callback function to observe training
    display(l)
    # plot current prediction against data
    if doplot
        pred = predict(state.u)
        pl = scatter(t, ode_data[1, :], label = "data")
        scatter!(pl, t, pred[1, :], label = "prediction")
        display(plot(pl))
    end
    return false
end
```

If the chosen method is a global optimizer that employs a local optimization
method, a similar set of common local optimizer arguments exists. Look at `MLSL` or `AUGLAG`
from NLopt for an example. The common local optimizer arguments are:

  - `local_method`: optimizer used for local optimization in global method
  - `local_maxiters`: the maximum number of iterations
  - `local_maxtime`: the maximum amount of time (in seconds) the optimization runs for
  - `local_abstol`: absolute tolerance in changes of the objective value
  - `local_reltol`: relative tolerance  in changes of the objective value
  - `local_options`: `NamedTuple` of keyword arguments for local optimizer
"""
function solve(prob::OptimizationProblem, alg, args...;
        kwargs...)::AbstractOptimizationSolution
    if supports_opt_cache_interface(alg)
        solve!(init(prob, alg, args...; kwargs...))
    else
        if prob.u0 !== nothing && !isconcretetype(eltype(prob.u0))
            throw(NonConcreteEltypeError(eltype(prob.u0)))
        end
        _check_opt_alg(prob, alg; kwargs...)
        __solve(prob, alg, args...; kwargs...)
    end
end

function SciMLBase.solve(
        prob::EnsembleProblem{T}, args...; kwargs...) where {T <: OptimizationProblem}
    return SciMLBase.__solve(prob, args...; kwargs...)
end

function _check_opt_alg(prob::OptimizationProblem, alg; kwargs...)
    !allowsbounds(alg) && (!isnothing(prob.lb) || !isnothing(prob.ub)) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) does not support box constraints. Either remove the `lb` or `ub` bounds passed to `OptimizationProblem` or use a different algorithm."))
    requiresbounds(alg) && isnothing(prob.lb) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) requires box constraints. Either pass `lb` and `ub` bounds to `OptimizationProblem` or use a different algorithm."))
    !allowsconstraints(alg) && !isnothing(prob.f.cons) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) does not support constraints. Either remove the `cons` function passed to `OptimizationFunction` or use a different algorithm."))
    requiresconstraints(alg) && isnothing(prob.f.cons) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) requires constraints, pass them with the `cons` kwarg in `OptimizationFunction`."))
    # Check that if constraints are present and the algorithm supports constraints, both lcons and ucons are provided
    allowsconstraints(alg) && !isnothing(prob.f.cons) &&
        (isnothing(prob.lcons) || isnothing(prob.ucons)) &&
        throw(ArgumentError("Constrained optimization problem requires both `lcons` and `ucons` to be provided to OptimizationProblem. " *
                            "Example: OptimizationProblem(optf, u0, p; lcons=[-Inf], ucons=[0.0])"))
    !allowscallback(alg) && haskey(kwargs, :callback) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) does not support callbacks, remove the `callback` keyword argument from the `solve` call."))
    requiresgradient(alg) && !(prob.f isa AbstractOptimizationFunction) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) requires gradients, hence use `OptimizationFunction` to generate them with an automatic differentiation backend e.g. `OptimizationFunction(f, AutoForwardDiff())` or pass it in with `grad` kwarg."))
    requireshessian(alg) && !(prob.f isa AbstractOptimizationFunction) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) requires hessians, hence use `OptimizationFunction` to generate them with an automatic differentiation backend e.g. `OptimizationFunction(f, AutoFiniteDiff(); kwargs...)` or pass them in with `hess` kwarg."))
    requiresconsjac(alg) && !(prob.f isa AbstractOptimizationFunction) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) requires constraint jacobians, hence use `OptimizationFunction` to generate them with an automatic differentiation backend e.g. `OptimizationFunction(f, AutoFiniteDiff(); kwargs...)` or pass them in with `cons` kwarg."))
    requiresconshess(alg) && !(prob.f isa AbstractOptimizationFunction) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) requires constraint hessians, hence use `OptimizationFunction` to generate them with an automatic differentiation backend e.g. `OptimizationFunction(f, AutoFiniteDiff(), AutoFiniteDiff(hess=true); kwargs...)` or pass them in with `cons` kwarg."))
    return
end

const OPTIMIZER_MISSING_ERROR_MESSAGE = """
                                        Optimization algorithm not found. Either the chosen algorithm is not a valid solver
                                        choice for the `OptimizationProblem`, or the Optimization solver library is not loaded.
                                        Make sure that you have loaded an appropriate Optimization.jl solver library, for example,
                                        `solve(prob,Optim.BFGS())` requires `using OptimizationOptimJL` and
                                        `solve(prob,Adam())` requires `using OptimizationOptimisers`.

                                        For more information, see the Optimization.jl documentation: https://docs.sciml.ai/Optimization/stable/.
                                        """

struct OptimizerMissingError <: Exception
    alg::Any
end

function Base.showerror(io::IO, e::OptimizerMissingError)
    println(io, OPTIMIZER_MISSING_ERROR_MESSAGE)
    print(io, "Chosen Optimizer: ")
    print(e.alg)
end

"""
```julia
init(prob::OptimizationProblem, alg::AbstractOptimizationAlgorithm, args...; kwargs...)
```

## Keyword Arguments

The arguments to `init` are the same as to `solve` and common across all of the optimizers.
These common arguments are:

  - `maxiters` (the maximum number of iterations)
  - `maxtime` (the maximum of time the optimization runs for)
  - `abstol` (absolute tolerance in changes of the objective value)
  - `reltol` (relative tolerance  in changes of the objective value)
  - `callback` (a callback function)

Some optimizer algorithms have special keyword arguments documented in the
solver portion of the documentation and their respective documentation.
These arguments can be passed as `kwargs...` to `init`.

See also [`solve(prob::OptimizationProblem, alg, args...; kwargs...)`](@ref)
"""
function init(prob::OptimizationProblem, alg, args...; kwargs...)::AbstractOptimizationCache
    if prob.u0 !== nothing && !isconcretetype(eltype(prob.u0))
        throw(NonConcreteEltypeError(eltype(prob.u0)))
    end
    _check_opt_alg(prob::OptimizationProblem, alg; kwargs...)
    cache = __init(prob, alg, args...; prob.kwargs..., kwargs...)
    return cache
end

"""
```julia
solve!(cache::AbstractOptimizationCache)
```

Solves the given optimization cache.

See also [`init(prob::OptimizationProblem, alg, args...; kwargs...)`](@ref)
"""
function solve!(cache::AbstractOptimizationCache)::AbstractOptimizationSolution
    __solve(cache)
end

# needs to be defined for each cache
supports_opt_cache_interface(alg) = false
function __solve(cache::AbstractOptimizationCache)::AbstractOptimizationSolution end
function __init(prob::OptimizationProblem, alg, args...;
        kwargs...)::AbstractOptimizationCache
    throw(OptimizerMissingError(alg))
end

# if no cache interface is supported at least the following method has to be defined
function __solve(prob::OptimizationProblem, alg, args...; kwargs...)
    throw(OptimizerMissingError(alg))
end


# Functions used in solve dispatches

eltypedual(x) = false
promote_u0(::Nothing, p, t0) = nothing
isdualtype(::Type{T}) where {T} = false

has_kwargs(_prob::AbstractSciMLProblem) = has_kwargs(typeof(_prob))
Base.@pure __has_kwargs(::Type{T}) where {T} = :kwargs ∈ fieldnames(T)
has_kwargs(::Type{T}) where {T} = __has_kwargs(T)

@inline function extract_alg(solve_args, solve_kwargs, prob_kwargs)
    if isempty(solve_args) || isnothing(first(solve_args))
        if haskey(solve_kwargs, :alg)
            solve_kwargs[:alg]
        elseif haskey(prob_kwargs, :alg)
            prob_kwargs[:alg]
        else
            nothing
        end
    elseif first(solve_args) isa SciMLBase.AbstractSciMLAlgorithm &&
           !(first(solve_args) isa SciMLBase.EnsembleAlgorithm)
        first(solve_args)
    else
        nothing
    end
end

handle_distribution_u0(_u0) = _u0

eval_u0(u0::Function) = true
eval_u0(u0) = false

function get_concrete_p(prob, kwargs)
    if haskey(kwargs, :p)
        p = kwargs[:p]
    else
        p = prob.p
    end
end

function get_concrete_u0(prob::BVProblem, isadapt, t0, kwargs)
    if haskey(kwargs, :u0)
        u0 = kwargs[:u0]
    else
        u0 = prob.u0
    end

    isadapt && eltype(u0) <: Integer && (u0 = float.(u0))

    _u0 = handle_distribution_u0(u0)

    if isinplace(prob) && (_u0 isa Number || _u0 isa SArray)
        throw(IncompatibleInitialConditionError())
    end

    if _u0 isa Tuple
        throw(TupleStateError())
    end

    return _u0
end

function checkkwargs(kwargshandle; kwargs...)
    if any(x -> x ∉ allowedkeywords, keys(kwargs))
        if kwargshandle == KeywordArgError
            throw(CommonKwargError(kwargs))
        elseif kwargshandle == KeywordArgWarn
            @warn KWARGWARN_MESSAGE
            unrecognized = setdiff(keys(kwargs), allowedkeywords)
            print("Unrecognized keyword arguments: ")
            printstyled(unrecognized; bold = true, color = :red)
            print("\n\n")
        else
            @assert kwargshandle == KeywordArgSilent
        end
    end
end

"""
    $(TYPEDSIGNATURES)

Given the index provider `indp` used to construct the problem `prob` being solved, return
an updated `prob` to be used for solving. All implementations should accept arbitrary
keyword arguments.

Should be called before the problem is solved, after performing type-promotion on the
problem. If the returned problem is not `===` the provided `prob`, it is assumed to
contain the `u0` and `p` passed as keyword arguments.

# Keyword Arguments

- `u0`, `p`: Override values for `state_values(prob)` and `parameter_values(prob)` which
  should be used instead of the ones in `prob`.
"""
function get_updated_symbolic_problem(indp, prob; kw...)
    return prob
end

function isconcreteu0(prob, t0, kwargs)
    !eval_u0(prob.u0) && prob.u0 !== nothing && !isdistribution(prob.u0)
end

function isconcretedu0(prob, t0, kwargs)
    !eval_u0(prob.u0) && prob.du0 !== nothing && !isdistribution(prob.du0)
end

function get_concrete_u0(prob, isadapt, t0, kwargs)
    if eval_u0(prob.u0)
        u0 = prob.u0(prob.p, t0)
    elseif haskey(kwargs, :u0)
        u0 = kwargs[:u0]
    else
        u0 = prob.u0
    end

    isadapt && eltype(u0) <: Integer && (u0 = float.(u0))

    _u0 = handle_distribution_u0(u0)

    if isinplace(prob) && (_u0 isa Number || _u0 isa SArray)
        throw(IncompatibleInitialConditionError())
    end

    nu0 = length(something(_u0, ()))
    if isdefined(prob.f, :mass_matrix) && prob.f.mass_matrix !== nothing &&
       prob.f.mass_matrix isa AbstractArray &&
       size(prob.f.mass_matrix, 1) !== nu0
        throw(IncompatibleMassMatrixError(size(prob.f.mass_matrix, 1), nu0))
    end

    if _u0 isa Tuple
        throw(TupleStateError())
    end

    _u0
end

function get_concrete_du0(prob, isadapt, t0, kwargs)
    if eval_u0(prob.du0)
        du0 = prob.du0(prob.p, t0)
    elseif haskey(kwargs, :du0)
        du0 = kwargs[:du0]
    else
        du0 = prob.du0
    end

    isadapt && eltype(du0) <: Integer && (du0 = float.(du0))

    _du0 = handle_distribution_u0(du0)

    if isinplace(prob) && (_du0 isa Number || _du0 isa SArray)
        throw(IncompatibleInitialConditionError())
    end

    _du0
end

function promote_u0(u0, p, t0)
    if SciMLStructures.isscimlstructure(p)
        _p = SciMLStructures.canonicalize(SciMLStructures.Tunable(), p)[1]
        if !isequal(_p, p)
            return promote_u0(u0, _p, t0)
        end
    end
    Tu = eltype(u0)
    if isdualtype(Tu)
        return u0
    end
    Tp = anyeltypedual(p, Val{0})
    if Tp == Any
        Tp = Tu
    end
    Tt = anyeltypedual(t0, Val{0})
    if Tt == Any
        Tt = Tu
    end
    Tcommon = promote_type(Tu, Tp, Tt)
    return if isdualtype(Tcommon)
        Tcommon.(u0)
    else
        u0
    end
end

function promote_u0(u0::AbstractArray{<:Complex}, p, t0)
    if SciMLStructures.isscimlstructure(p)
        _p = SciMLStructures.canonicalize(SciMLStructures.Tunable(), p)[1]
        if !isequal(_p, p)
            return promote_u0(u0, _p, t0)
        end
    end
    Tu = real(eltype(u0))
    if isdualtype(Tu)
        return u0
    end
    Tp = anyeltypedual(p, Val{0})
    if Tp == Any
        Tp = Tu
    end
    Tt = anyeltypedual(t0, Val{0})
    if Tt == Any
        Tt = Tu
    end
    Tcommon = promote_type(eltype(u0), Tp, Tt)
    return if isdualtype(real(Tcommon))
        Tcommon.(u0)
    else
        u0
    end
end

anyeltypedual(x) = anyeltypedual(x, Val{0})
anyeltypedual(x, counter) = Any
anyeltypedual(x::FixedSizeDiffCache, counter = 0) = Any

value(x) = x
unitfulvalue(x) = x
isdistribution(u0) = false
sse(x::Number) = abs2(x)

struct DualEltypeChecker{T, T2}
    x::T
    counter::T2
end

@inline __sum(f::F, args...; init, kwargs...) where {F} = sum(f, args...; init, kwargs...)
@inline function __sum(
        f::F, a::StaticArraysCore.StaticArray...; init, kwargs...) where {F}
    return mapreduce(f, +, a...; init, kwargs...)
end

totallength(x::Number) = 1
totallength(x::AbstractArray) = __sum(totallength, x; init = 0)

_reshape(v, siz) = reshape(v, siz)
_reshape(v::Number, siz) = v
_reshape(v::AbstractSciMLScalarOperator, siz) = v

set_mooncakeoriginator_if_mooncake(x::SciMLBase.ADOriginator) = x

# Copied from Static.jl https://github.com/SciML/Static.jl/blob/b50279cc9b33741fd60f382c789fbaef8622d964/src/Static.jl#L743
@generated function reduce_tup(f::F, inds::Tuple{Vararg{Any, N}}) where {F, N}
    q = Expr(:block, Expr(:meta, :inline, :propagate_inbounds))
    if N == 1
        push!(q.args, :(inds[1]))
        return q
    end
    syms = Vector{Symbol}(undef, N)
    i = 0
    for n in 1:N
        syms[n] = iₙ = Symbol(:i_, (i += 1))
        push!(q.args, Expr(:(=), iₙ, Expr(:ref, :inds, n)))
    end
    W = 1 << (8sizeof(N) - 2 - leading_zeros(N))
    while W > 0
        _N = length(syms)
        for _ in (2W):W:_N
            for w in 1:W
                new_sym = Symbol(:i_, (i += 1))
                push!(q.args, Expr(:(=), new_sym, Expr(:call, :f, syms[w], syms[w + W])))
                syms[w] = new_sym
            end
            deleteat!(syms, (1 + W):(2W))
        end
        W >>>= 1
    end
    q
end

####
# Catch undefined AD overload cases

const ADJOINT_NOT_FOUND_MESSAGE = """
                                  Compatibility with reverse-mode automatic differentiation requires SciMLSensitivity.jl.
                                  Please install SciMLSensitivity.jl and do `using SciMLSensitivity`/`import SciMLSensitivity`
                                  for this functionality. For more details, see https://sensitivity.sciml.ai/dev/.
                                  """

struct AdjointNotFoundError <: Exception end

function Base.showerror(io::IO, e::AdjointNotFoundError)
    print(io, ADJOINT_NOT_FOUND_MESSAGE)
    println(io, TruncatedStacktraces.VERBOSE_MSG)
end

function _concrete_solve_adjoint(args...; kwargs...)
    throw(AdjointNotFoundError())
end

const FORWARD_SENSITIVITY_NOT_FOUND_MESSAGE = """
                                              Compatibility with forward-mode automatic differentiation requires SciMLSensitivity.jl.
                                              Please install SciMLSensitivity.jl and do `using SciMLSensitivity`/`import SciMLSensitivity`
                                              for this functionality. For more details, see https://sensitivity.sciml.ai/dev/.
                                              """

struct ForwardSensitivityNotFoundError <: Exception end

function Base.showerror(io::IO, e::ForwardSensitivityNotFoundError)
    print(io, FORWARD_SENSITIVITY_NOT_FOUND_MESSAGE)
    println(io, TruncatedStacktraces.VERBOSE_MSG)
end

function _concrete_solve_forward(args...; kwargs...)
    throw(ForwardSensitivityNotFoundError())
end