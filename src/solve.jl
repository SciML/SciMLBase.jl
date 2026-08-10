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
heterogeneous values such as the `ArrayPartition`
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
    return print(io, e.eltype)
end

# Functions used in solve dispatches

eltypedual(x) = false
promote_u0(::Nothing, p, t0) = nothing
isdualtype(::Type{T}) where {T} = false

has_kwargs(_prob::AbstractSciMLProblem) = has_kwargs(typeof(_prob))
# Branched on in `solve`/`init` argument handling, so it has to fold at compile time.
@generated function __has_kwargs(::Type{T}) where {T}
    return :($(:kwargs ∈ fieldnames(T)))
end
has_kwargs(::Type{T}) where {T} = __has_kwargs(T)

@inline function extract_alg(solve_args, solve_kwargs, prob_kwargs)
    return if isempty(solve_args) || isnothing(first(solve_args))
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

"""
    get_concrete_p(prob, kwargs)

Return the parameter value a solver should use for a single solve call.

# Arguments

- `prob`: A SciML problem with a `p` field.
- `kwargs`: Keyword arguments from the solve call, represented by a `NamedTuple` or
  another key-addressable keyword container.

# Returns

The `p` keyword override when present; otherwise `prob.p`.

# Developer Interface

Solver packages call this before constructing their cache or concretizing a problem so
that `solve(prob; p = new_p)` and `solve(remake(prob; p = new_p))` use the same
parameter value. Extensions should preserve that override rule and must not mutate
`prob` or the supplied keyword container.
"""
function get_concrete_p(prob, kwargs)
    return if haskey(kwargs, :p)
        p = kwargs[:p]
    else
        p = prob.p
    end
end

"""
    get_concrete_u0(prob, isadapt, t0, kwargs)

Return the initial state a solver should use for a single solve call.

# Arguments

- `prob`: A SciML problem with a `u0` field.
- `isadapt`: Whether the solver will adapt its time step or mesh. Integer initial
  states are converted to floating-point values when this is `true`.
- `t0`: The initial independent-variable value.
- `kwargs`: Keyword arguments from the solve call. A `u0` entry overrides `prob.u0`.

# Returns

The effective initial state after applying the `u0` override, evaluating supported
problem-specific initial-state representations, and enforcing the common in-place and
tuple-state constraints.

# Developer Interface

Solver packages use this hook while concretizing a problem. Extensions must honor the
`u0` keyword override, return a state compatible with the problem's in-place trait,
and throw the appropriate SciMLBase initial-condition error instead of silently
changing an invalid state representation.
"""
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

function warn_controller_kwargs(unrecognized)
    return if any(in(controller_kwargs), unrecognized)
        printstyled(CONTROLLER_KWARG_MESSAGE; color = :cyan)
        print("\n")
    end
end

function checkkwargs(kwargshandle; kwargs...)
    return if any(x -> x ∉ allowedkeywords, keys(kwargs))
        if kwargshandle == KeywordArgError
            throw(CommonKwargError(kwargs))
        elseif kwargshandle == KeywordArgWarn
            @warn KWARGWARN_MESSAGE
            unrecognized = setdiff(keys(kwargs), allowedkeywords)
            print("Unrecognized keyword arguments: ")
            printstyled(unrecognized; bold = true, color = :red)
            print("\n\n")
            warn_controller_kwargs(unrecognized)
        else
            @assert kwargshandle == KeywordArgSilent
        end
    end
end

function checkkwargs(kwargshandle, allowed; kwargs...)
    return if any(x -> x ∉ allowed, keys(kwargs))
        if kwargshandle == KeywordArgError
            throw(CommonKwargError(kwargs))
        elseif kwargshandle == KeywordArgWarn
            @warn KWARGWARN_MESSAGE
            unrecognized = setdiff(keys(kwargs), allowed)
            print("Unrecognized keyword arguments: ")
            printstyled(unrecognized; bold = true, color = :red)
            print("\n\n")
            warn_controller_kwargs(unrecognized)
        else
            @assert kwargshandle == KeywordArgSilent
        end
    end
end
"""
    get_updated_symbolic_problem(indp, prob; kwargs...) -> updated_prob

Return the problem that a solver should use after applying symbolic solve-time updates.

# Arguments

- `indp`: The root index provider returned by [`get_root_indp`](@ref); this is the
  primary extension dispatch argument.
- `prob`: The type-promoted SciML problem about to be solved.

# Keywords

- `u0`: A solve-time state override, defaulting to the problem's state values.
- `p`: A solve-time parameter override, defaulting to the problem's parameter values.
- `kwargs...`: Additional solve keywords. Implementations must accept arbitrary keywords.

# Returns

- `updated_prob`: `prob` or a replacement problem ready for `init`/`solve`. When the
  result is not `=== prob`, it must already contain the effective `u0` and `p` values.

# Extension Rules

Symbolic-system packages may specialize on `indp` and problem types they own. This hook is
called after type promotion and before solver initialization. Implementations must preserve
the problem family and all solve-relevant fields not explicitly replaced.

# Example

```julia
struct MySolveSystem end
struct MySymbolicProblem
    u0
    p
end

function SciMLBase.get_updated_symbolic_problem(
        ::MySolveSystem, prob::MySymbolicProblem; u0 = prob.u0, p = prob.p, kwargs...)
    return MySymbolicProblem(u0, p)
end
```
"""
function get_updated_symbolic_problem(indp, prob; kw...)
    return prob
end

"""
    isconcreteu0(prob, t0, kwargs) -> Bool

Return whether `prob.u0` is already a concrete initial state that can be reused without
evaluation.

# Arguments

- `prob`: A SciML problem with a `u0` field.
- `t0`: The initial independent-variable value for the proposed solve.
- `kwargs`: Keyword arguments for the proposed solve.

# Returns

`true` when the problem's stored `u0` is neither deferred nor distribution-valued, and
`false` otherwise.

# Developer Interface

`get_concrete_problem` implementations use this predicate to decide whether they may
return the original problem object. Extensions must return `false` whenever evaluating
or replacing `u0` is required, including for solve-call overrides.
"""
function isconcreteu0(prob, t0, kwargs)
    return !eval_u0(prob.u0) && prob.u0 !== nothing && !isdistribution(prob.u0)
end

function isconcretedu0(prob, t0, kwargs)
    return !eval_u0(prob.u0) && prob.du0 !== nothing && !isdistribution(prob.du0)
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

    return _u0
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

    return _du0
end

"""
    promote_u0(u0, p, t0)

Promote an initial state to preserve automatic-differentiation element types carried by
parameters or the initial independent variable.

# Arguments

- `u0`: Initial state to prepare for a solve.
- `p`: Effective parameter value for the solve.
- `t0`: Effective initial independent-variable value.

# Returns

`u0` unchanged when no dual element type is present; otherwise a state with the common
dual-compatible element type.

# Developer Interface

Solver packages call this after `get_concrete_u0` and before constructing caches or
testing whether a problem can be reused. Extensions must retain the value semantics of
`u0` and only change its element type when promotion is required by `p` or `t0`.
"""
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
anyeltypedual(x::DespecializedParameters) = anyeltypedual(x.params)
anyeltypedual(x::DespecializedParameters, counter) = anyeltypedual(x.params, counter)
anyeltypedual(x::FixedSizeDiffCache, counter = 0) = Any

"""
    value(x)

Return the plain scalar or type representation underlying `x`.

Numeric-wrapper integrations may specialize this hook to remove AD,
uncertainty, or unit wrappers when a solver needs an ordinary numeric value for
control flow or type selection. The default returns `x` unchanged.

!!! warning "Developer API, not user API"
    Solver and numeric-wrapper packages may extend this hook. Application code
    should preserve its numeric wrappers instead of stripping them manually.

# Example
```julia
SciMLBase.value(x::MyTrackedNumber) = x.primal
```
"""
value(x) = x

"""
    unitfulvalue(x)

Return the numeric value of `x` while retaining its physical units.

Numeric-wrapper integrations may specialize this hook to remove AD or
uncertainty wrappers without discarding a unit carried by the primal value. The
default returns `x` unchanged. Use [`value`](@ref) when the solver instead needs
a fully unwrapped scalar or type.

!!! warning "Developer API, not user API"
    Solver and numeric-wrapper packages may extend this hook. Application code
    should use its quantity package's operations directly.

# Example
```julia
SciMLBase.unitfulvalue(x::MyDualQuantity) = x.primal
```
"""
unitfulvalue(x) = x
isdistribution(u0) = false
sse(x::Number) = abs2(x)

"""
    get_concrete_problem(prob, isadapt; alg = nothing, kwargs...)

Return the problem object a solver should use for a specific solve call.

# Arguments

- `prob`: The problem supplied to `solve` or `init`.
- `isadapt`: Whether the selected algorithm adapts time steps or a mesh.
- `alg`: Selected algorithm, when algorithm-dependent promotion or function
  specialization is required.
- `kwargs`: Solve-call keyword arguments, including possible `u0`, `p`, and time-span
  overrides.

# Returns

Either `prob` when its stored data already matches the requested solve, or a replacement
problem carrying the effective values for that solve.

# Developer Interface

Solver packages extend this hook for problem families that require solver-time
concretization. Implementations should use `get_concrete_p`, `get_concrete_u0`,
`promote_u0`, and `remake` as appropriate; they must not mutate `prob`, must preserve
the problem family and user-visible metadata, and may return `prob` only when the
effective values and their relevant types are unchanged.
"""
function get_concrete_problem end

"""
    check_prob_alg_pairing(prob, alg)

Validate that `alg` is applicable to `prob` before a solver allocates its cache.

# Arguments

- `prob`: Problem selected for the solve.
- `alg`: Algorithm selected for the solve.

# Returns

`nothing` when the pairing is supported.

# Developer Interface

Solver packages extend this hook for problem families with algorithm restrictions.
Implementations should throw a descriptive SciMLBase error for unsupported pairings and
must not mutate `prob` or `alg`. A no-op method is appropriate when every algorithm in
the package's documented algorithm family supports the problem type.
"""
function check_prob_alg_pairing end

struct DualEltypeChecker{T, T2}
    x::T
    counter::T2
end

@inline __sum(f::F, args...; init, kwargs...) where {F} = sum(f, args...; init, kwargs...)
@inline function __sum(
        f::F, a::StaticArraysCore.StaticArray...; init, kwargs...
    ) where {F}
    return mapreduce(f, +, a...; init, kwargs...)
end

totallength(x::Number) = 1
totallength(x::AbstractArray) = __sum(totallength, x; init = 0)

_reshape(v, siz) = reshape(v, siz)
_reshape(v::Number, siz) = v
_reshape(v::AbstractSciMLScalarOperator, siz) = v

"""
    set_mooncakeoriginator_if_mooncake(originator::ADOriginator)

Return the automatic-differentiation originator for a solver call, preserving
`originator` in ordinary execution and switching to `MooncakeOriginator()` when
Mooncake's overlay evaluates the call.

# Developer API

Solver and sensitivity packages pass a concrete `ADOriginator` through their
low-level solve path so AD rules can dispatch on its origin. End-user code
should not call this function or dispatch on its result.
"""
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
    return q
end

####
# Catch undefined AD overload cases

const ADJOINT_NOT_FOUND_MESSAGE = """
Compatibility with reverse-mode automatic differentiation requires SciMLSensitivity.jl.
Please install SciMLSensitivity.jl and do `using SciMLSensitivity`/`import SciMLSensitivity`
for this functionality. For more details, see <https://docs.sciml.ai/SciMLSensitivity/dev/>.
"""

struct AdjointNotFoundError <: Exception end

function Base.showerror(io::IO, e::AdjointNotFoundError)
    return print(io, ADJOINT_NOT_FOUND_MESSAGE)
end

"""
    _concrete_solve_adjoint(prob, alg, sensealg, u0, p, originator, args...; kwargs...)

Construct the reverse-mode derivative result for a solver call.

# Arguments

  - `prob`: The problem being solved.
  - `alg`: The selected solver algorithm, which may be `nothing` when the caller uses
    a problem-stored default.
  - `sensealg`: The selected sensitivity algorithm or `nothing` for a package-defined
    default.
  - `u0`: The effective initial state passed to the primal solve.
  - `p`: The effective parameter value passed to the primal solve.
  - `originator`: An [`ADOriginator`](@ref) identifying the outer AD system.
  - `args...`: Remaining positional solve arguments.

# Keyword Arguments

`kwargs...` are the solve keywords forwarded by the caller. Implementations must honor
the applicable common solve keywords and preserve any values that affect the primal
solution or derivative result.

# Returns

A pair `(primal, pullback)`. `primal` is the ordinary solve result and `pullback` is a
callable compatible with the AD system identified by `originator`.

# Developer Interface

Sensitivity packages extend this hook to implement reverse-mode solve derivatives.
Methods must specialize on at least one problem type, solver/sensitivity algorithm, or
originator type that the extending package owns. They must not mutate `prob`, `u0`, or
`p`, must compute the same primal result as the corresponding `solve` call, and must
return cotangents in the positional order expected by the originating AD rule. The
fallback throws an informative error until a compatible sensitivity package has loaded.
"""
function _concrete_solve_adjoint(args...; kwargs...)
    throw(AdjointNotFoundError())
end

const FORWARD_SENSITIVITY_NOT_FOUND_MESSAGE = """
Compatibility with forward-mode automatic differentiation requires SciMLSensitivity.jl.
Please install SciMLSensitivity.jl and do `using SciMLSensitivity`/`import SciMLSensitivity`
for this functionality. For more details, see <https://docs.sciml.ai/SciMLSensitivity/dev/>.
"""

struct ForwardSensitivityNotFoundError <: Exception end

function Base.showerror(io::IO, e::ForwardSensitivityNotFoundError)
    return print(io, FORWARD_SENSITIVITY_NOT_FOUND_MESSAGE)
end

"""
    _concrete_solve_forward(prob, alg, sensealg, u0, p, originator, args...; kwargs...)

Construct the forward-mode derivative result for a solver call.

# Arguments

  - `prob`: The problem being solved.
  - `alg`: The selected solver algorithm, which may be `nothing` when the caller uses
    a problem-stored default.
  - `sensealg`: The selected sensitivity algorithm or `nothing` for a package-defined
    default.
  - `u0`: The effective initial state passed to the primal solve.
  - `p`: The effective parameter value passed to the primal solve.
  - `originator`: An [`ADOriginator`](@ref) identifying the outer AD system.
  - `args...`: Remaining positional solve arguments.

# Keyword Arguments

`kwargs...` are the solve keywords forwarded by the caller. Implementations must honor
the applicable common solve keywords and preserve any values that affect the primal
solution or tangent result.

# Returns

A pair `(primal, pushforward)`. `primal` is the ordinary solve result and `pushforward`
is a callable compatible with the AD system identified by `originator`.

# Developer Interface

Sensitivity packages extend this hook to implement forward-mode solve derivatives.
Methods must specialize on at least one problem type, solver/sensitivity algorithm, or
originator type that the extending package owns. They must not mutate `prob`, `u0`, or
`p`, must compute the same primal result as the corresponding `solve` call, and must
accept tangents in the positional order expected by the originating AD rule. The
fallback throws an informative error until a compatible sensitivity package has loaded.
"""
function _concrete_solve_forward(args...; kwargs...)
    throw(ForwardSensitivityNotFoundError())
end
