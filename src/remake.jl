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
    TwoPointBVPFunction    # EnsembleProblem,
]
    @eval remaker_of(::$T) = $T
end

"""
    remake(thing; <keyword arguments>)

Re-construct `thing` with new field values specified by the keyword
arguments.
"""
function remake(thing; kwargs...)
    _remake_internal(thing; kwargs...)
end

function _remake_internal(thing; kwargs...)
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
    remake(prob::AbstractSciMLProblem; u0 = missing, p = missing, interpret_symbolicmap = true, use_defaults = false)

Remake the given problem `prob`. If `u0` or `p` are given, they will be used instead
of the unknowns/parameters of the problem. Either of them can be a symbolic map if
the problem has an associated system. If `interpret_symbolicmap == false`, `p` will never
be interpreted as a symbolic map and used as-is for parameters. `use_defaults` allows
controlling whether the default values from the system will be used to calculate missing
values in the symbolic map passed to `u0` or `p`. It is only valid when either `u0` or
`p` have been explicitly provided as a symbolic map and the problem has an associated
system.
"""
function remake(prob::AbstractSciMLProblem; u0 = missing,
        p = missing, interpret_symbolicmap = true, use_defaults = false, kwargs...)
    u0, p = updated_u0_p(prob, u0, p; interpret_symbolicmap, use_defaults)
    _remake_internal(prob; kwargs..., u0, p)
end

function remake(prob::AbstractIntervalNonlinearProblem; p = missing,
        interpret_symbolicmap = true, use_defaults = false, kwargs...)
    _, p = updated_u0_p(prob, [], p; interpret_symbolicmap, use_defaults)
    _remake_internal(prob; kwargs..., p)
end

function remake(prob::AbstractNoiseProblem; kwargs...)
    _remake_internal(prob; kwargs...)
end

function remake(
        prob::AbstractIntegralProblem; p = missing, interpret_symbolicmap = true, use_defaults = false, kwargs...)
    _, p = updated_u0_p(prob, nothing, p; interpret_symbolicmap, use_defaults)
    _remake_internal(prob; kwargs..., p)
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
        interpret_symbolicmap = true,
        use_defaults = false,
        _kwargs...)
    if tspan === missing
        tspan = prob.tspan
    end

    newu0, newp = updated_u0_p(prob, u0, p; interpret_symbolicmap, use_defaults)

    iip = isinplace(prob)

    if f === missing
        initializeprob, initializeprobmap = remake_initializeprob(
            prob.f.sys, prob.f, u0 === missing ? newu0 : u0,
            tspan[1], p === missing ? newp : p)
        if specialization(prob.f) === FunctionWrapperSpecialize
            ptspan = promote_tspan(tspan)
            if iip
                _f = ODEFunction{iip, FunctionWrapperSpecialize}(
                    wrapfun_iip(
                        unwrapped_f(prob.f.f),
                        (newu0, newu0, newp,
                            ptspan[1]));
                    initializeprob, initializeprobmap)
            else
                _f = ODEFunction{iip, FunctionWrapperSpecialize}(
                    wrapfun_oop(
                        unwrapped_f(prob.f.f),
                        (newu0, newp,
                            ptspan[1]));
                    initializeprob, initializeprobmap)
            end
        else
            _f = prob.f
            if __has_initializeprob(_f)
                props = getproperties(_f)
                @reset props.initializeprob = initializeprob
                props = values(props)
                _f = parameterless_type(_f){
                    iip, specialization(_f), map(typeof, props)...}(props...)
            end
            if __has_initializeprobmap(_f)
                props = getproperties(_f)
                @reset props.initializeprobmap = initializeprobmap
                props = values(props)
                _f = parameterless_type(_f){
                    iip, specialization(_f), map(typeof, props)...}(props...)
            end
        end
    elseif f isa AbstractODEFunction
        _f = f
    elseif specialization(prob.f) === FunctionWrapperSpecialize
        ptspan = promote_tspan(tspan)
        if iip
            _f = ODEFunction{iip, FunctionWrapperSpecialize}(wrapfun_iip(f,
                (newu0, newu0, newp,
                    ptspan[1])))
        else
            _f = ODEFunction{iip, FunctionWrapperSpecialize}(wrapfun_oop(f,
                (newu0, newp, ptspan[1])))
        end
    else
        _f = ODEFunction{isinplace(prob), specialization(prob.f)}(f)
    end

    if kwargs === missing
        ODEProblem{isinplace(prob)}(
            _f, newu0, tspan, newp, prob.problem_type; prob.kwargs...,
            _kwargs...)
    else
        ODEProblem{isinplace(prob)}(_f, newu0, tspan, newp, prob.problem_type; kwargs...)
    end
end

"""
    remake_initializeprob(sys, scimlfn, u0, t0, p)

Re-create the initialization problem present in the function `scimlfn`, using the
associated system `sys`, and the new values of `u0`, initial time `t0` and `p`. By
default, returns `nothing, nothing` if `scimlfn` does not have an initialization
problem, and `scimlfn.initializeprob, scimlfn.initializeprobmap` if it does.
"""
function remake_initializeprob(sys, scimlfn, u0, t0, p)
    if !has_initializeprob(scimlfn)
        return nothing, nothing
    end
    return scimlfn.initializeprob, scimlfn.initializeprobmap
end

"""
    remake(prob::BVProblem; f = missing, u0 = missing, tspan = missing,
           p = missing, kwargs = missing, problem_type = missing, _kwargs...)

Remake the given `BVProblem`.
"""
function remake(prob::BVProblem{uType, tType, iip, nlls}; f = missing, bc = missing,
        u0 = missing, tspan = missing, p = missing, kwargs = missing, problem_type = missing,
        interpret_symbolicmap = true, use_defaults = false, _kwargs...) where {
        uType, tType, iip, nlls}
    if tspan === missing
        tspan = prob.tspan
    end

    u0, p = updated_u0_p(prob, u0, p; interpret_symbolicmap, use_defaults)

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
            _f = BVPFunction{iip, FunctionWrapperSpecialize, twopoint}(
                wrapfun_iip(f,
                    (u0, u0, p, ptspan[1])), bc; prob.f.bcresid_prototype)
        else
            _f = BVPFunction{iip, FunctionWrapperSpecialize, twopoint}(
                wrapfun_oop(f,
                    (u0, p, ptspan[1])), bc; prob.f.bcresid_prototype)
        end
    else
        _f = BVPFunction{isinplace(prob), specialization(prob.f), twopoint}(f, bc;
            prob.f.bcresid_prototype)
    end

    if kwargs === missing
        BVProblem{iip}(
            _f, bc, u0, tspan, p; problem_type, nlls = Val(nlls), prob.kwargs...,
            _kwargs...)
    else
        BVProblem{iip}(_f, bc, u0, tspan, p; problem_type, nlls = Val(nlls), kwargs...)
    end
end

"""
    remake(prob::SDEProblem; f = missing, g = missing, u0 = missing, tspan = missing,
           p = missing, noise = missing, noise_rate_prototype = missing,
           seed = missing, kwargs = missing, _kwargs...)

Remake the given `SDEProblem`.
"""
function remake(prob::SDEProblem;
        f = missing,
        g = missing,
        u0 = missing,
        tspan = missing,
        p = missing,
        noise = missing,
        noise_rate_prototype = missing,
        interpret_symbolicmap = true,
        use_defaults = false,
        seed = missing,
        kwargs = missing,
        _kwargs...)
    if tspan === missing
        tspan = prob.tspan
    end

    u0, p = updated_u0_p(prob, u0, p; interpret_symbolicmap, use_defaults)

    if noise === missing
        noise = prob.noise
    end

    if noise_rate_prototype === missing
        noise_rate_prototype = prob.noise_rate_prototype
    end

    if seed === missing
        seed = prob.seed
    end

    if f === missing && g === missing
        f = prob.f
        g = prob.g
    elseif f !== missing && g === missing
        g = prob.g
    elseif f === missing && g !== missing
        if prob.f isa SDEFunction
            f = remake(prob.f; g = g)
        else
            f = SDEFunction(prob.f, g)
        end
    else
        if f isa SDEFunction
            f = remake(f; g = g)
        else
            f = SDEFunction(f, g)
        end
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
    remake(func::SDEFunction; f = missing, g = missing,
           mass_matrix = missing, analytic = missing, kwargs...)

Remake the given `SDEFunction`.
"""
function remake(func::SDEFunction;
        f = missing,
        g = missing,
        mass_matrix = missing,
        analytic = missing,
        kwargs...)
    if f === missing
        f = func.f
    end

    if g === missing
        g = func.g
    end

    if mass_matrix === missing
        mass_matrix = func.mass_matrix
    end

    if analytic === missing
        analytic = func.analytic
    end

    return SDEFunction(f, g; mass_matrix, analytic, kwargs...)
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
        interpret_symbolicmap = true,
        use_defaults = false,
        _kwargs...)
    u0, p = updated_u0_p(prob, u0, p; interpret_symbolicmap, use_defaults)
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
        interpret_symbolicmap = true,
        use_defaults = false,
        _kwargs...)
    u0, p = updated_u0_p(prob, u0, p; interpret_symbolicmap, use_defaults)
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
        interpret_symbolicmap = true, use_defaults = false, kwargs = missing, _kwargs...)
    u0, p = updated_u0_p(prob, u0, p; interpret_symbolicmap, use_defaults)

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

function varmap_has_var(varmap, var)
    haskey(varmap, var) || hasname(var) && haskey(varmap, getname(var))
end

function varmap_get(varmap, var, default = nothing)
    if haskey(varmap, var)
        return varmap[var]
    end
    if hasname(var)
        name = getname(var)
        if haskey(varmap, name)
            return varmap[name]
        end
    end
    return default
end

anydict(d) = Dict{Any, Any}(d)
anydict() = Dict{Any, Any}()

function _updated_u0_p_internal(
        prob, ::Missing, p; interpret_symbolicmap = true, use_defaults = false)
    u0 = state_values(prob)

    if p isa AbstractArray && isempty(p)
        return _updated_u0_p_internal(
            prob, u0, parameter_values(prob); interpret_symbolicmap)
    end
    eltype(p) <: Pair && interpret_symbolicmap || return u0, p
    defs = default_values(prob)
    p = fill_p(prob, anydict(p); defs, use_defaults)
    return _updated_u0_p_symmap(prob, u0, Val(false), p, Val(true))
end

function _updated_u0_p_internal(
        prob, u0, ::Missing; interpret_symbolicmap = true, use_defaults = false)
    p = parameter_values(prob)

    eltype(u0) <: Pair || return u0, p
    defs = default_values(prob)
    u0 = fill_u0(prob, anydict(u0); defs, use_defaults)
    return _updated_u0_p_symmap(prob, u0, Val(true), p, Val(false))
end

function _updated_u0_p_internal(
        prob, u0, p; interpret_symbolicmap = true, use_defaults = false)
    isu0symbolic = eltype(u0) <: Pair
    ispsymbolic = eltype(p) <: Pair && interpret_symbolicmap

    if !isu0symbolic && !ispsymbolic
        return u0, p
    end
    defs = default_values(prob)
    if isu0symbolic
        u0 = fill_u0(prob, anydict(u0); defs, use_defaults)
    end
    if ispsymbolic
        p = fill_p(prob, anydict(p); defs, use_defaults)
    end
    return _updated_u0_p_symmap(prob, u0, Val(isu0symbolic), p, Val(ispsymbolic))
end

function fill_u0(prob, u0; defs = nothing, use_defaults = false)
    vsyms = variable_symbols(prob)
    if length(u0) == length(vsyms)
        return u0
    end
    newvals = anydict()
    for sym in vsyms
        varmap_has_var(u0, sym) && continue
        def = if defs === nothing || (defval = varmap_get(defs, sym)) === nothing ||
                 (symbolic_type(defval) === NotSymbolic() && !use_defaults)
            nothing
        else
            defval
        end
        newvals[sym] = @something def getu(prob, sym)(prob)
    end
    return merge(u0, newvals)
end

function fill_p(prob, p; defs = nothing, use_defaults = false)
    psyms = parameter_symbols(prob)::Vector
    if length(p) == length(psyms)
        return p
    end
    newvals = anydict()
    for sym in psyms
        varmap_has_var(p, sym) && continue
        def = if defs === nothing || (defval = varmap_get(defs, sym)) === nothing ||
                 (symbolic_type(defval) === NotSymbolic() && !use_defaults)
            nothing
        else
            defval
        end
        newvals[sym] = @something def getp(prob, sym)(prob)
    end
    return merge(p, newvals)
end

function _updated_u0_p_symmap(prob, u0, ::Val{true}, p, ::Val{false})
    isdep = any(symbolic_type(v) !== NotSymbolic() for (_, v) in u0)
    isdep || return remake_buffer(prob, state_values(prob), u0), p

    u0 = anydict(k => symbolic_type(v) === NotSymbolic() ? v : symbolic_evaluate(v, u0)
    for (k, v) in u0)

    isdep = any(symbolic_type(v) !== NotSymbolic() for (_, v) in u0)
    isdep || return remake_buffer(prob, state_values(prob), u0), p

    # FIXME: need to provide `u` since the observed function expects it.
    # This is sort of an implicit dependency on MTK. The values of `u` won't actually be
    # used, since any state symbols in the expression were substituted out earlier.
    temp_state = ProblemState(; u = state_values(prob), p = p)
    u0 = anydict(k => symbolic_type(v) === NotSymbolic() ? v : getu(prob, v)(temp_state)
    for (k, v) in u0)
    return remake_buffer(prob, state_values(prob), u0), p
end

function _updated_u0_p_symmap(prob, u0, ::Val{false}, p, ::Val{true})
    isdep = any(symbolic_type(v) !== NotSymbolic() for (_, v) in p)
    isdep || return u0, remake_buffer(prob, parameter_values(prob), p)

    p = anydict(k => symbolic_type(v) === NotSymbolic() ? v : symbolic_evaluate(v, p)
    for (k, v) in p)

    isdep = any(symbolic_type(v) !== NotSymbolic() for (_, v) in p)
    isdep || return u0, remake_buffer(prob, parameter_values(prob), p)

    # FIXME: need to provide `p` since the observed function expects an `MTKParameters`
    # this is sort of an implicit dependency on MTK. The values of `p` won't actually be
    # used, since any parameter symbols in the expression were substituted out earlier.
    temp_state = ProblemState(; u = u0, p = parameter_values(prob))
    p = anydict(k => symbolic_type(v) === NotSymbolic() ? v : getu(prob, v)(temp_state)
    for (k, v) in p)
    return u0, remake_buffer(prob, parameter_values(prob), p)
end

function _updated_u0_p_symmap(prob, u0, ::Val{true}, p, ::Val{true})
    isu0dep = any(symbolic_type(v) !== NotSymbolic() for (_, v) in u0)
    ispdep = any(symbolic_type(v) !== NotSymbolic() for (_, v) in p)

    if !isu0dep && !ispdep
        return remake_buffer(prob, state_values(prob), u0),
        remake_buffer(prob, parameter_values(prob), p)
    end
    if !isu0dep
        u0 = remake_buffer(prob, state_values(prob), u0)
        return _updated_u0_p_symmap(prob, u0, Val(false), p, Val(true))
    end
    if !ispdep
        p = remake_buffer(prob, parameter_values(prob), p)
        return _updated_u0_p_symmap(prob, u0, Val(true), p, Val(false))
    end

    varmap = merge(u0, p)
    u0 = anydict(k => symbolic_type(v) === NotSymbolic() ? v : symbolic_evaluate(v, varmap)
    for (k, v) in u0)
    p = anydict(k => symbolic_type(v) === NotSymbolic() ? v : symbolic_evaluate(v, varmap)
    for (k, v) in p)
    return remake_buffer(prob, state_values(prob), u0),
    remake_buffer(prob, parameter_values(prob), p)
end

function updated_u0_p(prob, u0, p; interpret_symbolicmap = true, use_defaults = false)
    if u0 === missing && p === missing
        return state_values(prob), parameter_values(prob)
    end
    if !has_sys(prob.f)
        if interpret_symbolicmap && eltype(p) !== Union{} && eltype(p) <: Pair
            throw(ArgumentError("This problem does not support symbolic maps with " *
                                "`remake`, i.e. it does not have a symbolic origin. Please use `remake`" *
                                "with the `p` keyword argument as a vector of values (paying attention to" *
                                "parameter order) or pass `interpret_symbolicmap = false` as a keyword argument"))
        end
        if eltype(u0) !== Union{} && eltype(u0) <: Pair
            throw(ArgumentError("This problem does not support symbolic maps with" *
                                " remake, i.e. it does not have a symbolic origin. Please use `remake`" *
                                "with the `u0` keyword argument as a vector of values, paying attention to the order."))
        end
        return (u0 === missing ? state_values(prob) : u0),
        (p === missing ? parameter_values(prob) : p)
    end
    return _updated_u0_p_internal(prob, u0, p; interpret_symbolicmap, use_defaults)
end

# overloaded in MTK to intercept symbolic remake
function process_p_u0_symbolic(prob, p, u0)
    if prob isa Union{AbstractDEProblem, OptimizationProblem, NonlinearProblem}
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
