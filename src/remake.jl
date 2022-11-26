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

"""
    remake(prob::ODEProblem; f = missing, u0 = missing, tspan = missing, 
           p = missing, kwargs = missing, _kwargs...)

Remake the given `ODEProblem`.
If `u0` or `p` are given as symbolic maps `ModelingToolkit.jl` has to be loaded.
"""
function remake(prob::ODEProblem; f = missing,
                u0 = missing,
                tspan = missing,
                p = missing,
                kwargs = missing,
                _kwargs...)
    if tspan === missing
        tspan = prob.tspan
    end

    if p === missing && u0 === missing
        p, u0 = prob.p, prob.u0
    else # at least one of them has a value
        if p === missing
            p = prob.p
        end
        if u0 === missing
            u0 = prob.u0
        end
        if (eltype(p) <: Pair && !isempty(p)) || (eltype(u0) <: Pair && !isempty(u0)) # one is a non-empty symbolic map
            hasproperty(prob.f, :sys) && hasfield(typeof(prob.f.sys), :ps) ||
                throw(ArgumentError("This problem does not support symbolic maps with `remake`, i.e. it does not have a symbolic origin." *
                                    " Please use `remake` with the `p` keyword argument as a vector of values, paying attention to parameter order."))
            hasproperty(prob.f, :sys) && hasfield(typeof(prob.f.sys), :states) ||
                throw(ArgumentError("This problem does not support symbolic maps with `remake`, i.e. it does not have a symbolic origin." *
                                    " Please use `remake` with the `u0` keyword argument as a vector of values, paying attention to state order."))
            p, u0 = process_p_u0_symbolic(prob, p, u0)
        end
    end

    iip = isinplace(prob)

    if f === missing
        if specialization(prob.f) === FunctionWrapperSpecialize
            ptspan = promote_tspan(tspan)
            if iip
                _f = ODEFunction{iip, FunctionWrapperSpecialize}(wrapfun_iip(unwrapped_f(prob.f.f),
                                                                             (u0, u0, p,
                                                                              ptspan[1])))
            else
                _f = ODEFunction{iip, FunctionWrapperSpecialize}(wrapfun_oop(unwrapped_f(prob.f.f),
                                                                             (u0, p,
                                                                              ptspan[1])))
            end
        else
            _f = prob.f
        end
    elseif f isa AbstractODEFunction
        _f = f
    elseif specialization(prob.f) === FunctionWrapperSpecialize
        ptspan = promote_tspan(tspan)
        if iip
            _f = ODEFunction{iip, FunctionWrapperSpecialize}(wrapfun_iip(f,
                                                                         (u0, u0, p,
                                                                          ptspan[1])))
        else
            _f = ODEFunction{iip, FunctionWrapperSpecialize}(wrapfun_oop(f,
                                                                         (u0, p, ptspan[1])))
        end
    else
        _f = ODEFunction{isinplace(prob), specialization(prob.f)}(f)
    end

    if kwargs === missing
        ODEProblem{isinplace(prob)}(_f, u0, tspan, p, prob.problem_type; prob.kwargs...,
                                    _kwargs...)
    else
        ODEProblem{isinplace(prob)}(_f, u0, tspan, p, prob.problem_type; kwargs...)
    end
end

# overloaded in MTK to intercept symbolic remake
function process_p_u0_symbolic(prob, p, u0)
    throw(ArgumentError("Please load `ModelingToolkit.jl` in order to support symbolic remake."))
end

function remake(thing::AbstractJumpProblem; kwargs...)
    parameterless_type(thing)(remake(thing.prob; kwargs...))
end

function remake(thing::AbstractEnsembleProblem; kwargs...)
    T = parameterless_type(thing)
    en_kwargs = [k for k in kwargs if first(k) ∈ fieldnames(T)]
    T(remake(thing.prob; setdiff(kwargs, en_kwargs)...); en_kwargs...)
end
