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
    if tspan === missing
        tspan = prob.tspan
    end

    if (p !== missing && eltype(p) <: Pair) || (u0 !== missing && eltype(u0) <: Pair)
        defs = Dict{Any, Any}()
        if hasproperty(prob.f, :sys)
            if hasfield(typeof(prob.f.sys), :ps)
                defs = mergedefaults(defs, prob.p, :parameters; sys=prob.f.sys)
            end
            if hasfield(typeof(prob.f.sys), :states)
                defs = mergedefaults(defs, prob.u0, :states; sys=prob.f.sys)
            end
        end
    else
        defs = nothing
    end

    if p === missing
        p = prob.p
    else
        if eltype(p) <: Pair
            if hasproperty(prob.f, :sys) && hasfield(typeof(prob.f.sys), :ps)
                p = handle_varmap(p, prob.f.sys, var_type = :parameters, defaults = defs)
                defs = mergedefaults(defs, p, :parameters; sys=prob.f.sys)
                @assert length(p) == length(prob.p)
            else
                throw(ArgumentError("This problem does not support symbolic parameter maps with `remake`, i.e. it does not have a symbolic origin. Please use `remake` with the `p` keyword argument as a vector of values, paying attention to parameter order."))
            end
        end
    end

    if u0 === missing
        u0 = prob.u0
    else
        if eltype(u0) <: Pair
            if hasproperty(prob.f, :sys) && hasfield(typeof(prob.f.sys), :states)
                u0 = handle_varmap(u0, prob.f.sys; var_type = :states,
                                   defaults = defs, tofloat = true)
                @assert length(u0) == length(prob.u0)
            else
                throw(ArgumentError("This problem does not support symbolic default maps with `remake`, i.e. it does not have a symbolic origin. Please use `remake` with the `u0` keyword argument as a vector of values, paying attention to the order of states."))
            end
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

function remake(thing::AbstractJumpProblem; kwargs...)
    parameterless_type(thing)(remake(thing.prob; kwargs...))
end

function remake(thing::AbstractEnsembleProblem; kwargs...)
    T = parameterless_type(thing)
    en_kwargs = [k for k in kwargs if first(k) ∈ fieldnames(T)]
    T(remake(thing.prob; setdiff(kwargs, en_kwargs)...); en_kwargs...)
end
