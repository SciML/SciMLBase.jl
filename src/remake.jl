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

"""
    remake(prob::BVProblem; f = missing, u0 = missing, tspan = missing,
           p = missing, kwargs = missing, problem_type = missing, _kwargs...)

Remake the given `BVProblem`.
"""
function remake(prob::BVProblem; f = missing, bc = missing, u0 = missing, tspan = missing,
    p = missing, kwargs = missing, problem_type = missing, _kwargs...)
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
    end

    iip = isinplace(prob)

    if problem_type === missing
        problem_type = prob.problem_type
    end

    twopoint = problem_type isa TwoPointBVProblem

    if bc === missing
        bc = prob.f.bc
    end

    if f === missing
        _f = prob.f
    elseif f isa BVPFunction
        _f = f
        bc = f.bc
    elseif specialization(prob.f) === FunctionWrapperSpecialize
        ptspan = promote_tspan(tspan)
        if iip
            _f = BVPFunction{iip, FunctionWrapperSpecialize, twopoint}(wrapfun_iip(f,
                (u0, u0, p, ptspan[1])), bc; prob.f.bcresid_prototype)
        else
            _f = BVPFunction{iip, FunctionWrapperSpecialize, twopoint}(wrapfun_oop(f,
                (u0, p, ptspan[1])), bc; prob.f.bcresid_prototype)
        end
    else
        _f = BVPFunction{isinplace(prob), specialization(prob.f), twopoint}(f, bc;
            prob.f.bcresid_prototype)
    end

    if kwargs === missing
        BVProblem{iip}(_f, bc, u0, tspan, p; problem_type, prob.kwargs..., _kwargs...)
    else
        BVProblem{iip}(_f, bc, u0, tspan, p; problem_type, kwargs...)
    end
end

"""
    remake(prob::SDEProblem; f = missing, u0 = missing, tspan = missing,
           p = missing, noise = missing, noise_rate_prototype = missing,
           seed = missing, kwargs = missing, _kwargs...)

Remake the given `SDEProblem`.
"""
function remake(prob::SDEProblem;
    f = missing,
    u0 = missing,
    tspan = missing,
    p = missing,
    noise = missing,
    noise_rate_prototype = missing,
    seed = missing,
    kwargs = missing,
    _kwargs...)
    if tspan === missing
        tspan = prob.tspan
    end

    if p === missing
        p = prob.p
    end

    if u0 === missing
        u0 = prob.u0
    end

    if noise === missing
        noise = prob.noise
    end

    if noise_rate_prototype === missing
        noise_rate_prototype = prob.noise_rate_prototype
    end

    if seed === missing
        seed = prob.seed
    end

    if f === missing #TODO: Need more features, e.g. remake `g`
        f = prob.f
    end

    iip = isinplace(prob)

    if kwargs === missing
        SDEProblem{iip}(f,
            u0,
            tspan,
            p;
            noise,
            noise_rate_prototype,
            seed,
            prob.kwargs...,
            _kwargs...)
    else
        SDEProblem{iip}(f, u0, tspan, p; noise, noise_rate_prototype, seed, kwargs...)
    end
end

"""
    remake(prob::OptimizationProblem; f = missing, u0 = missing, p = missing,
        lb = missing, ub = missing, int = missing, lcons = missing, ucons = missing,
        sense = missing, kwargs = missing, _kwargs...)

Remake the given `OptimizationProblem`.
If `u0` or `p` are given as symbolic maps `ModelingToolkit.jl` has to be loaded.
"""
function remake(prob::OptimizationProblem;
    f = missing,
    u0 = missing,
    p = missing,
    lb = missing,
    ub = missing,
    int = missing,
    lcons = missing,
    ucons = missing,
    sense = missing,
    kwargs = missing,
    _kwargs...)
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

    if f === missing
        f = prob.f
    end
    if lb === missing
        lb = prob.lb
    end
    if ub === missing
        ub = prob.ub
    end
    if int === missing
        int = prob.int
    end
    if lcons === missing
        lcons = prob.lcons
    end
    if ucons === missing
        ucons = prob.ucons
    end
    if sense === missing
        sense = prob.sense
    end

    if kwargs === missing
        OptimizationProblem{isinplace(prob)}(f = f, u0 = u0, p = p, lb = lb,
            ub = ub, int = int,
            lcons = lcons, ucons = ucons,
            sense = sense; prob.kwargs..., _kwargs...)
    else
        OptimizationProblem{isinplace(prob)}(f = f, u0 = u0, p = p, lb = lb,
            ub = ub, int = int,
            lcons = lcons, ucons = ucons,
            sense = sense; kwargs...)
    end
end

"""
    remake(prob::NonlinearProblem; f = missing, u0 = missing, p = missing,
        problem_type = missing, kwargs = missing, _kwargs...)

Remake the given `NonlinearProblem`.
If `u0` or `p` are given as symbolic maps `ModelingToolkit.jl` has to be loaded.
"""
function remake(prob::NonlinearProblem;
    f = missing,
    u0 = missing,
    p = missing,
    problem_type = missing,
    kwargs = missing,
    _kwargs...)
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

    if f === missing
        f = prob.f
    end
    if problem_type === missing
        problem_type = prob.problem_type
    end

    if kwargs === missing
        NonlinearProblem{isinplace(prob)}(f = f, u0 = u0, p = p,
            problem_type = problem_type; prob.kwargs...,
            _kwargs...)
    else
        NonlinearProblem{isinplace(prob)}(f = f, u0 = u0, p = p,
            problem_type = problem_type; kwargs...)
    end
end


"""
    remake(prob::NonlinearLeastSquaresProblem; f = missing, u0 = missing, p = missing,
        kwargs = missing, _kwargs...)

Remake the given `NonlinearLeastSquaresProblem`.
"""
function remake(prob::NonlinearLeastSquaresProblem; f = missing, u0 = missing, p = missing,
    kwargs = missing, _kwargs...)
    if p === missing && u0 === missing
        p, u0 = prob.p, prob.u0
    else # at least one of them has a value
        if p === missing
            p = prob.p
        end
        if u0 === missing
            u0 = prob.u0
        end
    end

    if f === missing
        f = prob.f
    end

    if kwargs === missing
        return NonlinearLeastSquaresProblem{isinplace(prob)}(; f, u0, p, prob.kwargs...,
            _kwargs...)
    else
        return NonlinearLeastSquaresProblem{isinplace(prob)}(; f, u0, p, kwargs...)
    end
end

# overloaded in MTK to intercept symbolic remake
function process_p_u0_symbolic(prob, p, u0)
    if typeof(prob) <: Union{AbstractDEProblem, OptimizationProblem, NonlinearProblem}
        throw(ArgumentError("Please load `ModelingToolkit.jl` in order to support symbolic remake."))
    else
        throw(ArgumentError("Symbolic remake for $(typeof(prob)) is currently not supported, consider opening an issue."))
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
