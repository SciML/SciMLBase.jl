@generated function struct_as_namedtuple(st)
    A = (Expr(:(=), n, :(st.$n)) for n in setdiff(fieldnames(st), (:kwargs,)))
    Expr(:tuple, A...)
end

Base.@pure function remaker_of(prob::T) where {T <: AbstractSciMLProblem}
    parameterless_type(T){isinplace(prob)}
end
Base.@pure remaker_of(alg::T) where {T} = parameterless_type(T)

# Define `remaker_of` for the types that does not (make sense to)
# implement `isinplace` trait:
for T in [
    NoiseProblem,
    SplitFunction,  # TODO: use isinplace path for type-stability
    TwoPointBVPFunction,
    # EnsembleProblem,
]
    @eval remaker_of(::$T) = $T
end

"""
    remake(thing; <keyword arguments>)

Re-construct `thing` with new field values specified by the keyword
arguments.
"""
function remake(thing; kwargs...)
    T = remaker_of(thing)
    if :kwargs ∈ fieldnames(typeof(thing))
        if :kwargs ∉ keys(kwargs)
            T(; struct_as_namedtuple(thing)..., thing.kwargs..., kwargs...)
        else
            T(; struct_as_namedtuple(thing)..., kwargs[:kwargs]...)
        end
    else
        T(; struct_as_namedtuple(thing)..., kwargs...)
    end
end

function isrecompile(prob::ODEProblem{iip}) where {iip}
    (prob.f isa ODEFunction) ? !isfunctionwrapper(prob.f.f) : true
end

function remake(prob::ODEProblem; f = missing,
                u0 = missing,
                tspan = missing,
                p = missing,
                kwargs = missing,
                _kwargs...)
    if u0 === missing
        u0 = prob.u0
    end

    if tspan === missing
        tspan = prob.tspan
    end

    if p === missing
        p = prob.p
    end

    if f === missing
        if prob.f isa ODEFunction && isinplace(prob) &&
           typeof(u0) <: Vector{Float64} &&
           eltype(promote_tspan(tspan)) <: Float64 &&
           typeof(p) <: Union{SciMLBase.NullParameters, Vector{Float64}}
            # If it's possible to FunctionWrapperSpecialize then do it
            _f = ODEFunction{isinplace(prob), FunctionWrapperSpecialize}(unwrapped_f(prob.f))
        elseif specialization(f) === FunctionWrapperSpecialize
            # It would FunctionWrapperSpecialize on types which are not allowed
            # Thus don't allow it and full specialize
            _f = ODEFunction{isinplace(prob), FullSpecialize}(unwrapped_f(prob.f))
        else
            # Otherwise just use the previous specialization choice
            # This would preserve no-specialize for those using it
            _f = ODEFunction{isinplace(prob), specialization(prob.f)}(unwrapped_f(prob.f))
        end
    elseif prob.f isa ODEFunction && isinplace(prob) &&
           typeof(u0) <: Vector{Float64} &&
           eltype(promote_tspan(tspan)) <: Float64 &&
           typeof(p) <: Union{SciMLBase.NullParameters, Vector{Float64}}
        _f = ODEFunction{isinplace(prob), FunctionWrapperSpecialize}(f)
    elseif prob.f isa ODEFunction
        _f = ODEFunction{isinplace(prob)}(f)
    else
        _f = f
    end

    if kwargs === missing
        ODEProblem{isinplace(prob)}(_f, u0, tspan, p, prob.problem_type; prob.kwargs...,
                                    _kwargs...)
    else
        ODEProblem{isinplace(prob)}(_f, u0, tspan, p, prob.problem_type; kwargs...)
    end
end

function remake(thing::AbstractJumpProblem; kwargs...)
    parameterless_type(thing)(remake(thing.prob; kwargs...))
end

function remake(thing::AbstractEnsembleProblem; kwargs...)
    T = parameterless_type(thing)
    en_kwargs = [k for k in kwargs if first(k) ∈ fieldnames(T)]
    T(remake(thing.prob; setdiff(kwargs, en_kwargs)...); en_kwargs...)
end
