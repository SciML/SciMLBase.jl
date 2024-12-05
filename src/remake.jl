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
    named_thing = struct_as_namedtuple(thing)
    if :kwargs ∈ fieldnames(typeof(thing))
        if :args ∈ fieldnames(typeof(thing))
            named_thing = Base.structdiff(named_thing, (; args = ()))
            if :args ∉ keys(kwargs)
                k = Base.structdiff(named_thing, (; args = ()))
                if :kwargs ∉ keys(kwargs)
                    T(; named_thing..., thing.kwargs..., kwargs...)
                else
                    T(; named_thing..., kwargs[:kwargs]...)
                end
            else
                kwargs2 = Base.structdiff((; kwargs...), (; args = ()))
                if :kwargs ∉ keys(kwargs)
                    T(kwargs[:args]...; named_thing..., thing.kwargs..., kwargs2...)
                else
                    T(kwargs[:args]...; named_thing..., kwargs2[:kwargs]...)
                end
            end
        else
            if :kwargs ∉ keys(kwargs)
                T(; named_thing..., thing.kwargs..., kwargs...)
            else
                T(; named_thing..., kwargs[:kwargs]...)
            end
        end
    else
        T(; named_thing..., kwargs...)
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
        build_initializeprob = true,
        use_defaults = false,
        lazy_initialization = nothing,
        _kwargs...)
    if tspan === missing
        tspan = prob.tspan
    end

    newu0, newp = updated_u0_p(prob, u0, p, tspan[1]; interpret_symbolicmap, use_defaults)

    iip = isinplace(prob)

    initialization_data = prob.f.initialization_data

    if f === missing
        if build_initializeprob
            initialization_data = remake_initialization_data_compat_wrapper(
                prob.f.sys, prob.f, u0, tspan[1], p, newu0, newp)
        else
            initialization_data = nothing
        end
        if specialization(prob.f) === FunctionWrapperSpecialize
            ptspan = promote_tspan(tspan)
            if iip
                _f = ODEFunction{iip, FunctionWrapperSpecialize}(
                    wrapfun_iip(
                        unwrapped_f(prob.f.f),
                        (newu0, newu0, newp,
                            ptspan[1])); initialization_data)
            else
                _f = ODEFunction{iip, FunctionWrapperSpecialize}(
                    wrapfun_oop(
                        unwrapped_f(prob.f.f),
                        (newu0, newp,
                            ptspan[1])); initialization_data)
            end
        else
            _f = prob.f
            if __has_initialization_data(_f)
                props = getproperties(_f)
                @reset props.initialization_data = initialization_data
                props = values(props)
                _f = parameterless_type(_f){iip, specialization(_f), map(typeof, props)...}(props...)
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

    prob = if kwargs === missing
        ODEProblem{isinplace(prob)}(
            _f, newu0, tspan, newp, prob.problem_type; prob.kwargs...,
            _kwargs...)
    else
        ODEProblem{isinplace(prob)}(_f, newu0, tspan, newp, prob.problem_type; kwargs...)
    end

    if lazy_initialization === nothing
        lazy_initialization = !is_trivial_initialization(initialization_data)
    end
    if !lazy_initialization
        u0, p, _ = get_initial_values(
            prob, prob, prob.f, OverrideInit(), Val(isinplace(prob)))
        if u0 !== nothing && eltype(u0) == Any && isempty(u0)
            u0 = nothing
        end
        @reset prob.u0 = u0
        @reset prob.p = p
    end

    return prob
end

"""
    remake_initializeprob(sys, scimlfn, u0, t0, p)

!! WARN
This method is deprecated. Please see `remake_initialization_data`

Re-create the initialization problem present in the function `scimlfn`, using the
associated system `sys`, and the user-provided new values of `u0`, initial time `t0` and
`p`. By default, returns `nothing, nothing, nothing, nothing` if `scimlfn` does not have an
initialization problem, and
`scimlfn.initializeprob, scimlfn.update_initializeprob!, scimlfn.initializeprobmap, scimlfn.initializeprobpmap`
if it does.

Note that `u0` or `p` may be `missing` if the user does not provide a value for them.
"""
function remake_initializeprob(sys, scimlfn, u0, t0, p)
    if !has_initialization_data(scimlfn)
        return nothing, nothing, nothing, nothing
    end
    initdata = scimlfn.initialization_data
    return initdata.initializeprob, initdata.update_initializeprob!,
    initdata.initializeprobmap, initdata.initializeprobpmap
end

"""
    $(TYPEDSIGNATURES)

Wrapper around `remake_initialization_data` for backward compatibility when `newu0` and
`newp` were not arguments.
"""
function remake_initialization_data_compat_wrapper(sys, scimlfn, u0, t0, p, newu0, newp)
    if hasmethod(remake_initialization_data,
        Tuple{typeof(sys), typeof(scimlfn), typeof(u0), typeof(t0), typeof(p)})
        remake_initialization_data(sys, scimlfn, u0, t0, p)
    else
        remake_initialization_data(sys, scimlfn, u0, t0, p, newu0, newp)
    end
end

"""
    remake_initialization_data(sys, scimlfn, u0, t0, p, newu0, newp)

Re-create the initialization data present in the function `scimlfn`, using the
associated system `sys`, the user provided new values of `u0`, initial time `t0`,
user-provided `p`, new u0 vector `newu0` and new parameter object `newp`. By default,
this calls `remake_initializeprob` for backward compatibility and attempts to construct
an `OverrideInitData` from the result.

Note that `u0` or `p` may be `missing` if the user does not provide a value for them.
"""
function remake_initialization_data(sys, scimlfn, u0, t0, p, newu0, newp)
    return reconstruct_initialization_data(
        nothing, remake_initializeprob(sys, scimlfn, u0, t0, p)...)
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

    u0, p = updated_u0_p(prob, u0, p, tspan[1]; interpret_symbolicmap, use_defaults)

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
        lazy_initialization = nothing,
        build_initializeprob = true,
        _kwargs...)
    if tspan === missing
        tspan = prob.tspan
    end

    newu0, newp = updated_u0_p(prob, u0, p, tspan[1]; interpret_symbolicmap, use_defaults)

    if build_initializeprob
        initialization_data = remake_initialization_data(
            prob.f.sys, prob.f, u0, tspan[1], p, newu0, newp)
    else
        initialization_data = nothing
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
    f = coalesce(f, prob.f)
    g = coalesce(g, prob.g)
    f = remake(prob.f; f, g, initialization_data)
    iip = isinplace(prob)

    prob = if kwargs === missing
        SDEProblem{iip}(f,
            newu0,
            tspan,
            newp;
            noise,
            noise_rate_prototype,
            seed,
            prob.kwargs...,
            _kwargs...)
    else
        SDEProblem{iip}(f, newu0, tspan, newp; noise, noise_rate_prototype, seed, kwargs...)
    end
    if lazy_initialization === nothing
        lazy_initialization = !is_trivial_initialization(initialization_data)
    end
    if !lazy_initialization
        u0, p, _ = get_initial_values(
            prob, prob, prob.f, OverrideInit(), Val(isinplace(prob)))
        if u0 !== nothing && eltype(u0) == Any && isempty(u0)
            u0 = nothing
        end
        @reset prob.u0 = u0
        @reset prob.p = p
    end

    return prob
end

"""
    remake(func::SDEFunction; f = missing, g = missing,
           mass_matrix = missing, analytic = missing, kwargs...)

Remake the given `SDEFunction`.
"""
function remake(func::Union{SDEFunction, SDDEFunction};
        f = missing,
        g = missing,
        mass_matrix = missing,
        analytic = missing,
        sys = missing,
        kwargs...)
    props = getproperties(func)
    props = @delete props.f
    props = @delete props.g
    @reset props.mass_matrix = coalesce(mass_matrix, func.mass_matrix)
    @reset props.analytic = coalesce(analytic, func.analytic)
    @reset props.sys = coalesce(sys, func.sys)

    if f === missing
        f = func.f
    end

    if g === missing
        g = func.g
    end

    if f isa AbstractSciMLFunction
        f = f.f
    end

    T = func isa SDEFunction ? SDEFunction : SDDEFunction
    return T{isinplace(func)}(f, g; props..., kwargs...)
end

function remake(prob::DDEProblem; f = missing, h = missing, u0 = missing,
        tspan = missing, p = missing, constant_lags = missing,
        dependent_lags = missing, order_discontinuity_t0 = missing,
        neutral = missing, kwargs = missing, interpret_symbolicmap = true,
        use_defaults = false, lazy_initialization = nothing, build_initializeprob = true,
        _kwargs...)
    if tspan === missing
        tspan = prob.tspan
    end

    newu0, newp = updated_u0_p(prob, u0, p, tspan[1]; interpret_symbolicmap, use_defaults)

    if build_initializeprob
        initialization_data = remake_initialization_data(
            prob.f.sys, prob.f, u0, tspan[1], p, newu0, newp)
    else
        initialization_data = nothing
    end

    f = coalesce(f, prob.f)
    f = remake(prob.f; f, initialization_data)

    h = coalesce(h, prob.h)
    constant_lags = coalesce(constant_lags, prob.constant_lags)
    dependent_lags = coalesce(dependent_lags, prob.dependent_lags)
    order_discontinuity_t0 = coalesce(order_discontinuity_t0, prob.order_discontinuity_t0)
    neutral = coalesce(neutral, prob.neutral)

    iip = isinplace(prob)

    prob = if kwargs === missing
        DDEProblem{iip}(f,
            newu0,
            h,
            tspan,
            newp;
            constant_lags,
            dependent_lags,
            order_discontinuity_t0,
            neutral,
            prob.kwargs...,
            _kwargs...)
    else
        DDEProblem{iip}(f, newu0, h, tspan, newp; constant_lags, dependent_lags,
            order_discontinuity_t0, neutral, kwargs...)
    end
    if lazy_initialization === nothing
        lazy_initialization = !is_trivial_initialization(initialization_data)
    end
    if !lazy_initialization
        u0, p, _ = get_initial_values(
            prob, prob, prob.f, OverrideInit(), Val(isinplace(prob)))
        if u0 !== nothing && eltype(u0) == Any && isempty(u0)
            u0 = nothing
        end
        @reset prob.u0 = u0
        @reset prob.p = p
    end

    return prob
end

function remake(func::DDEFunction;
        f = missing,
        mass_matrix = missing,
        analytic = missing,
        sys = missing,
        kwargs...)
    props = getproperties(func)
    props = @delete props.f
    @reset props.mass_matrix = coalesce(mass_matrix, func.mass_matrix)
    @reset props.analytic = coalesce(analytic, func.analytic)
    @reset props.sys = coalesce(sys, func.sys)

    if f === missing
        f = func.f
    end
    if f isa AbstractSciMLFunction
        f = f.f
    end

    return DDEFunction{isinplace(func)}(f; props..., kwargs...)
end

function remake(prob::SDDEProblem;
        f = missing,
        g = missing,
        h = missing,
        u0 = missing,
        tspan = missing,
        p = missing,
        constant_lags = missing,
        dependent_lags = missing,
        order_discontinuity_t0 = missing,
        neutral = missing,
        noise = missing,
        noise_rate_prototype = missing,
        interpret_symbolicmap = true,
        use_defaults = false,
        seed = missing,
        kwargs = missing,
        lazy_initialization = nothing,
        build_initializeprob = true,
        _kwargs...)
    if tspan === missing
        tspan = prob.tspan
    end

    newu0, newp = updated_u0_p(prob, u0, p, tspan[1]; interpret_symbolicmap, use_defaults)

    if build_initializeprob
        initialization_data = remake_initialization_data(
            prob.f.sys, prob.f, u0, tspan[1], p, newu0, newp)
    else
        initialization_data = nothing
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

    f = coalesce(f, prob.f)
    g = coalesce(g, prob.g)
    f = remake(prob.f; f, g, initialization_data)
    iip = isinplace(prob)

    h = coalesce(h, prob.h)
    constant_lags = coalesce(constant_lags, prob.constant_lags)
    dependent_lags = coalesce(dependent_lags, prob.dependent_lags)
    order_discontinuity_t0 = coalesce(order_discontinuity_t0, prob.order_discontinuity_t0)
    neutral = coalesce(neutral, prob.neutral)

    prob = if kwargs === missing
        SDDEProblem{iip}(f,
            g,
            newu0,
            h,
            tspan,
            newp;
            noise,
            noise_rate_prototype,
            seed,
            constant_lags,
            dependent_lags,
            order_discontinuity_t0,
            neutral,
            prob.kwargs...,
            _kwargs...)
    else
        SDDEProblem{iip}(
            f, g, newu0, tspan, newp; noise, noise_rate_prototype, seed, constant_lags,
            dependent_lags, order_discontinuity_t0, neutral, kwargs...)
    end

    if lazy_initialization === nothing
        lazy_initialization = !is_trivial_initialization(initialization_data)
    end
    if !lazy_initialization
        u0, p, _ = get_initial_values(
            prob, prob, prob.f, OverrideInit(), Val(isinplace(prob)))
        if u0 !== nothing && eltype(u0) == Any && isempty(u0)
            u0 = nothing
        end
        @reset prob.u0 = u0
        @reset prob.p = p
    end

    return prob
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

"""
    remake(prob::SCCNonlinearProblem; u0 = missing, p = missing, probs = missing,
        parameters_alias = prob.parameters_alias, sys = missing, explicitfuns! = missing)

Remake the given `SCCNonlinearProblem`. `u0` is the state vector for the entire problem,
which will be chunked appropriately and used to `remake` the individual subproblems. `p`
is the parameter object for `prob`. If `parameters_alias`, the same parameter object will be
used to `remake` the individual subproblems. Otherwise if `p !== missing`, this function will
error and require that `probs` be specified. `probs` is the collection of subproblems. Even if
`probs` is explicitly specified, the value of `u0` provided to `remake` will be used to
override the values in `probs`. `sys` is the index provider for the full system.
"""
function remake(prob::SCCNonlinearProblem; u0 = missing, p = missing, probs = missing,
        parameters_alias = prob.parameters_alias, sys = missing,
        interpret_symbolicmap = true, use_defaults = false, explicitfuns! = missing)
    if p !== missing && !parameters_alias && probs === missing
        throw(ArgumentError("`parameters_alias` is `false` for the given `SCCNonlinearProblem`. Please provide the subproblems using the keyword `probs` with the parameters updated appropriately in each."))
    end
    newu0, newp = updated_u0_p(prob, u0, p; interpret_symbolicmap, use_defaults)
    if probs === missing
        probs = prob.probs
    end
    offset = 0
    if u0 !== missing || p !== missing && parameters_alias
        probs = map(probs) do subprob
            subprob = if parameters_alias
                remake(subprob;
                    u0 = newu0[(offset + 1):(offset + length(state_values(subprob)))],
                    p = newp)
            else
                remake(subprob;
                    u0 = newu0[(offset + 1):(offset + length(state_values(subprob)))])
            end
            offset += length(state_values(subprob))
            return subprob
        end
    end
    if sys === missing
        sys = prob.f.sys
    end
    return SCCNonlinearProblem(
        probs, explicitfuns!, newp, parameters_alias; sys)
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

"""
    $(TYPEDSIGNATURES)

Check if `varmap::Dict{Any, Any}` contains cyclic values for any symbolic variables in
`syms`. Falls back on the basis of `symbolic_container(indp)`. Returns `false` by default.
"""
function detect_cycles(indp, varmap, syms)
    if hasmethod(symbolic_container, Tuple{typeof(indp)}) &&
       (sc = symbolic_container(indp)) != indp
        return detect_cycles(sc, varmap, syms)
    else
        return false
    end
end

anydict(d::Dict{Any, Any}) = d
anydict(d) = Dict{Any, Any}(d)
anydict() = Dict{Any, Any}()

function _updated_u0_p_internal(
        prob, ::Missing, ::Missing, t0; interpret_symbolicmap = true, use_defaults = false)
    return state_values(prob), parameter_values(prob)
end
function _updated_u0_p_internal(
        prob, ::Missing, p, t0; interpret_symbolicmap = true, use_defaults = false)
    u0 = state_values(prob)

    if p isa AbstractArray && isempty(p)
        return _updated_u0_p_internal(
            prob, u0, parameter_values(prob), t0; interpret_symbolicmap)
    end
    eltype(p) <: Pair && interpret_symbolicmap || return u0, p
    defs = default_values(prob)
    p = fill_p(prob, anydict(p); defs, use_defaults)
    return _updated_u0_p_symmap(prob, u0, Val(false), p, Val(true), t0)
end

function _updated_u0_p_internal(
        prob, u0, ::Missing, t0; interpret_symbolicmap = true, use_defaults = false)
    p = parameter_values(prob)

    eltype(u0) <: Pair || return u0, p
    defs = default_values(prob)
    u0 = fill_u0(prob, anydict(u0); defs, use_defaults)
    return _updated_u0_p_symmap(prob, u0, Val(true), p, Val(false), t0)
end

function _updated_u0_p_internal(
        prob, u0, p, t0; interpret_symbolicmap = true, use_defaults = false)
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
    return _updated_u0_p_symmap(prob, u0, Val(isu0symbolic), p, Val(ispsymbolic), t0)
end

function fill_u0(prob, u0; defs = nothing, use_defaults = false)
    fill_vars(prob, u0; defs, use_defaults, allsyms = variable_symbols(prob),
        index_function = variable_index)
end

function fill_p(prob, p; defs = nothing, use_defaults = false)
    fill_vars(prob, p; defs, use_defaults, allsyms = parameter_symbols(prob),
        index_function = parameter_index)
end

function fill_vars(
        prob, varmap; defs = nothing, use_defaults = false, allsyms, index_function)
    idx_to_vsym = anydict(index_function(prob, sym) => sym for sym in allsyms)
    sym_to_idx = anydict()
    idx_to_sym = anydict()
    idx_to_val = anydict()
    for (k, v) in varmap
        v === nothing && continue
        idx = index_function(prob, k)
        idx === nothing && continue
        if !(idx isa AbstractArray) || symbolic_type(k) != ArraySymbolic()
            idx = (idx,)
            k = (k,)
            v = (v,)
        end
        for (kk, vv, ii) in zip(k, v, idx)
            sym_to_idx[kk] = ii
            kk = idx_to_vsym[ii]
            sym_to_idx[kk] = ii
            idx_to_sym[ii] = kk
            idx_to_val[ii] = vv
        end
    end
    for sym in allsyms
        haskey(sym_to_idx, sym) && continue
        idx = index_function(prob, sym)
        haskey(idx_to_val, idx) && continue
        sym_to_idx[sym] = idx
        idx_to_sym[idx] = sym
        idx_to_val[idx] = if defs !== nothing &&
                             (defval = varmap_get(defs, sym)) !== nothing &&
                             (symbolic_type(defval) != NotSymbolic() || use_defaults)
            defval
        else
            getsym(prob, sym)(prob)
        end
    end
    newvals = anydict()
    for (idx, val) in idx_to_val
        newvals[idx_to_sym[idx]] = val
    end
    for (k, v) in varmap
        haskey(sym_to_idx, k) && continue
        newvals[k] = v
    end
    return newvals
end

struct CyclicDependencyError <: Exception
    varmap::Dict{Any, Any}
    vars::Any
end

function Base.showerror(io::IO, err::CyclicDependencyError)
    println(io, "Detected cyclic dependency in initial values:")
    for (k, v) in err.varmap
        println(io, k, " => ", "v")
    end
    println(io, "While trying to solve for variables: ", err.vars)
end

function _updated_u0_p_symmap(prob, u0, ::Val{true}, p, ::Val{false}, t0)
    isdep = any(symbolic_type(v) !== NotSymbolic() for (_, v) in u0)
    isdep || return remake_buffer(prob, state_values(prob), keys(u0), values(u0)), p

    if detect_cycles(prob, u0, variable_symbols(prob))
        throw(CyclicDependencyError(u0, variable_symbols(prob)))
    end
    for (k, v) in u0
        u0[k] = symbolic_type(v) === NotSymbolic() ? v : symbolic_evaluate(v, u0)
    end

    isdep = any(symbolic_type(v) !== NotSymbolic() for (_, v) in u0)
    isdep || return remake_buffer(prob, state_values(prob), keys(u0), values(u0)), p

    # FIXME: need to provide `u` since the observed function expects it.
    # This is sort of an implicit dependency on MTK. The values of `u` won't actually be
    # used, since any state symbols in the expression were substituted out earlier.
    temp_state = ProblemState(; u = state_values(prob), p = p, t = t0,
        h = is_markovian(prob) ? nothing : get_history_function(prob))
    for (k, v) in u0
        u0[k] = symbolic_type(v) === NotSymbolic() ? v : getsym(prob, v)(temp_state)
    end
    return remake_buffer(prob, state_values(prob), keys(u0), values(u0)), p
end

function _updated_u0_p_symmap(prob, u0, ::Val{false}, p, ::Val{true}, t0)
    isdep = any(symbolic_type(v) !== NotSymbolic() for (_, v) in p)
    isdep || return u0, remake_buffer(prob, parameter_values(prob), keys(p), values(p))

    if detect_cycles(prob, p, parameter_symbols(prob))
        throw(CyclicDependencyError(p, parameter_symbols(prob)))
    end
    for (k, v) in p
        p[k] = symbolic_type(v) === NotSymbolic() ? v : symbolic_evaluate(v, p)
    end

    isdep = any(symbolic_type(v) !== NotSymbolic() for (_, v) in p)
    isdep || return u0, remake_buffer(prob, parameter_values(prob), keys(p), values(p))

    # FIXME: need to provide `p` since the observed function expects an `MTKParameters`
    # this is sort of an implicit dependency on MTK. The values of `p` won't actually be
    # used, since any parameter symbols in the expression were substituted out earlier.
    temp_state = ProblemState(; u = u0, p = parameter_values(prob), t = t0,
        h = is_markovian(prob) ? nothing : get_history_function(prob))
    for (k, v) in p
        p[k] = symbolic_type(v) === NotSymbolic() ? v : getsym(prob, v)(temp_state)
    end
    return u0, remake_buffer(prob, parameter_values(prob), keys(p), values(p))
end

function _updated_u0_p_symmap(prob, u0, ::Val{true}, p, ::Val{true}, t0)
    isu0dep = any(symbolic_type(v) !== NotSymbolic() for (_, v) in u0)
    ispdep = any(symbolic_type(v) !== NotSymbolic() for (_, v) in p)

    if !isu0dep && !ispdep
        return remake_buffer(prob, state_values(prob), keys(u0), values(u0)),
        remake_buffer(prob, parameter_values(prob), keys(p), values(p))
    end

    varmap = merge(u0, p)
    allsyms = [variable_symbols(prob); parameter_symbols(prob)]
    if detect_cycles(prob, varmap, allsyms)
        throw(CyclicDependencyError(varmap, allsyms))
    end
    if is_time_dependent(prob)
        varmap[only(independent_variable_symbols(prob))] = t0
    end
    for (k, v) in u0
        u0[k] = symbolic_type(v) === NotSymbolic() ? v : symbolic_evaluate(v, varmap)
    end
    for (k, v) in p
        p[k] = symbolic_type(v) === NotSymbolic() ? v : symbolic_evaluate(v, varmap)
    end
    return remake_buffer(prob, state_values(prob), keys(u0), values(u0)),
    remake_buffer(prob, parameter_values(prob), keys(p), values(p))
end

function updated_u0_p(
        prob, u0, p, t0 = nothing; interpret_symbolicmap = true,
        use_defaults = false)
    if u0 === missing && p === missing
        return state_values(prob), parameter_values(prob)
    end
    if has_sys(prob.f) && prob.f.sys === nothing
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
    return _updated_u0_p_internal(prob, u0, p, t0; interpret_symbolicmap, use_defaults)
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
