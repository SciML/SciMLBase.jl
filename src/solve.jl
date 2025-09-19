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
                                  for this functionality. For more details, see <https://docs.sciml.ai/SciMLSensitivity/dev/>.
                                  """

struct AdjointNotFoundError <: Exception end

function Base.showerror(io::IO, e::AdjointNotFoundError)
    print(io, ADJOINT_NOT_FOUND_MESSAGE)
end

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
    print(io, FORWARD_SENSITIVITY_NOT_FOUND_MESSAGE)
end

function _concrete_solve_forward(args...; kwargs...)
    throw(ForwardSensitivityNotFoundError())
end
