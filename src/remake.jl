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
    if f === missing
        if prob.f isa ODEFunction &&
           prob.f.f isa FunctionWrappersWrappers.FunctionWrappersWrapper &&
           ((u0 !== missing && !(typeof(u0) <: Vector{Float64})) ||
            (tspan !== missing && !(eltype(promote_tspan(tspan)) <: Float64)) ||
            (p !== missing &&
             !(typeof(p) <: Union{SciMLBase.NullParameters, Vector{Float64}})))
            _f = ODEFunction{isinplace(prob)}(unwrapped_f(prob.f))
        else
            _f = prob.f
        end
    elseif prob.f isa ODEFunction # avoid the SplitFunction etc. cases
        _f = ODEFunction{isinplace(prob)}(f)
    elseif !isrecompile(prob)
        if isinplace(prob)
            _f = wrapfun_iip(unwrap_fw(f), (u0, u0, p, tspan[1]))
        else
            _f = wrapfun_oop(unwrap_fw(f), (u0, p, tspan[1]))
        end
        _f = ODEFunction{isinplace(prob)}(f)
    else
        _f = f
    end

    if u0 === missing
        u0 = prob.u0
    end

    if tspan === missing
        tspan = prob.tspan
    end

    if p === missing
        p = prob.p
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
